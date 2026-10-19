"""Qwen3.5 GDN cache recipe: attention KV and recurrent state."""

from __future__ import annotations

from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.configs.base import (
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import (
    LinearAttnConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
    cache_dtype_name,
    mxfp8_kv_scale_fields,
    scatter_stored_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    split_recurrent_state_groups,
)
from tokenspeed.runtime.utils.env import envs

_QWEN_GDN_PREFIX_GRANULARITY = 128


class QwenGDNRecipe(CacheRecipe):
    """Qwen3.5 full attention interleaved with GDN state.

    Draft (MTP) layers are full-attention continuation layers of the one big
    model, so they join the target's full-attention group.
    """

    family = "qwen_gdn"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.attn_config.kv_cache_mxfp8:
            raise RuntimeError(
                "Qwen cache buffer does not yet support the MXFP8 interleaved "
                "scale layout"
            )
        linear_attn = self.attn_config.component(LinearAttnConfig)
        if linear_attn is None:
            raise ValueError("Qwen GDN cache requires a linear-attention component")
        # The GDN backend reads the same decision, so publish it once here
        # rather than as a side effect of sizing the workspace.
        linear_attn.replay_ssm = self.replay_ssm

    # ---- layer vocabulary ----

    @cached_property
    def target_layer_types(self) -> tuple[str, ...]:
        return tuple(self.attn_config.component(SoftmaxAttnConfig).cache_layer_types)

    @cached_property
    def layer_types(self) -> tuple[str, ...]:
        return self.target_layer_types + (FULL_ATTENTION,) * self.num_draft_layers

    @cached_property
    def group_ids(self) -> tuple[str, ...]:
        # Recurrent state splits into its own groups; draft layers land in the
        # target's full-attention group.
        return (
            tuple(split_recurrent_state_groups(self.target_layer_types))
            + (FULL_ATTENTION,) * self.num_draft_layers
        )

    # ---- model geometry ----

    @property
    @override
    def prefix_granularity(self) -> int:
        return _QWEN_GDN_PREFIX_GRANULARITY

    @property
    @override
    def max_padding_fraction(self) -> float:
        """Allow the extra full-attention planes used by Qwen MTP drafts."""
        num_full_attention = sum(
            layer_type == FULL_ATTENTION for layer_type in self.target_layer_types
        )
        if num_full_attention == 0:
            raise ValueError("Qwen3.5 cache requires at least one full-attention layer")
        return 1.0 + 2.0 * self.num_draft_layers / num_full_attention

    # ---- capacity: sum state + history parents (not flat full-attn packing) ----

    @override
    def num_lcm_blocks(self, layout: CacheLayout) -> int:
        """Parents demanded by all cache groups, capped by the byte budget.

        Flat ``token_limit // (full_packing * P)`` under-counts GDN: each
        linear-attention state group needs its own LCM parents, and the
        scheduler admits against the *sum* of per-group demand.
        """
        usable_bytes = self.cache_budget_bytes - self.workspace_bytes()
        budgeted = self._budgeted_parents(usable_bytes, layout.lcm_block_bytes)
        if self.token_limit is None:
            return budgeted
        return min(budgeted, self.parents_needed(layout, self.token_limit))

    @override
    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        upper = self.token_limit
        if upper is None:
            full_packing = dict(layout.group_packing)[FULL_ATTENTION]
            upper = num_lcm_blocks * full_packing * layout.prefix_granularity
        return self._capacity_from_parents(layout, num_lcm_blocks, upper_bound=upper)

    # ---- decoder-layer fields ----

    @cached_property
    def _state_shapes(self):
        """Per-rank GDN state shapes from the linear component.

        Dtypes are the recipe's contract: conv bf16, SSM per the engine env.
        """
        linear_attn = self.attn_config.component(LinearAttnConfig)
        ssm_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[envs.TOKENSPEED_MAMBA_SSM_DTYPE.get()]
        return (
            linear_attn.conv_state_shape,
            cache_dtype_name(torch.bfloat16),
            linear_attn.temporal_state_shape,
            cache_dtype_name(ssm_dtype),
        )

    @cached_property
    def _kv_shape(self) -> tuple[int, ...]:
        spec = self.attn_config.component(SoftmaxAttnConfig)
        return (
            self.prefix_granularity,
            max(spec.num_kv_heads // spec.attn_tp_size, 1),
            spec.head_dim,
        )

    @cached_property
    def _draft_kv_shape(self) -> tuple[int, ...]:
        spec = self.draft_attn_config.component(SoftmaxAttnConfig)
        return (
            self.prefix_granularity,
            max(spec.num_kv_heads // spec.attn_tp_size, 1),
            spec.head_dim,
        )

    @override
    def fields_for_layer(
        self, layer_id: int, group_id: str, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        if layer_id >= len(self.target_layer_types):
            return self._draft_fields(layer_id, occurrence)
        conv_shape, conv_dtype, ssm_shape, ssm_dtype = self._state_shapes
        if self.layer_types[layer_id] == LINEAR_ATTENTION:
            return (
                CacheFieldSpec(
                    f"layer.{layer_id}.ssm",
                    f"unit.{occurrence}.a",
                    ssm_shape,
                    ssm_dtype,
                ),
                CacheFieldSpec(
                    f"layer.{layer_id}.conv",
                    f"unit.{occurrence}.b",
                    conv_shape,
                    conv_dtype,
                    exact_page_stride=False,
                ),
            )
        kv_dtype = scatter_stored_dtype_name(self.attn_config.kv_cache_dtype)
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.k", f"unit.{occurrence}.a", self._kv_shape, kv_dtype
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.v", f"unit.{occurrence}.b", self._kv_shape, kv_dtype
            ),
        )

    def _draft_fields(
        self, layer_id: int, occurrence: int
    ) -> tuple[CacheFieldSpec, ...]:
        """One MTP draft layer's full-attention KV, mxfp8 scales included."""
        config = self.draft_attn_config
        mxfp8 = bool(config.kv_cache_mxfp8)
        kv_dtype = (
            cache_dtype_name(torch.float8_e4m3fn)
            if mxfp8
            else scatter_stored_dtype_name(config.kv_cache_dtype)
        )
        fields = (
            CacheFieldSpec(
                f"layer.{layer_id}.k",
                f"unit.{occurrence}.a",
                self._draft_kv_shape,
                kv_dtype,
            ),
            CacheFieldSpec(
                f"layer.{layer_id}.v",
                f"unit.{occurrence}.b",
                self._draft_kv_shape,
                kv_dtype,
            ),
        )
        if not mxfp8:
            return fields
        return fields + mxfp8_kv_scale_fields(
            layer_id=layer_id,
            occurrence=occurrence,
            kv_heads=self._draft_kv_shape[1],
            head_dim=config.component(SoftmaxAttnConfig).head_dim,
            prefix_granularity=self.prefix_granularity,
        )

    # ---- speculative verify workspace ----

    @cached_property
    def replay_ssm(self) -> bool:
        """Whether the GDN backend replays the SSM state instead of staging it."""
        if not self.num_draft_layers:
            return False
        if not (
            self.server_args.enable_replay_ssm
            and int(self.server_args.speculative_num_draft_tokens) > 1
            and torch.device(self.attn_config.device).type == "cuda"
        ):
            return False
        from tokenspeed_kernel.ops.attention import gdn_replay_commit_supported

        return bool(gdn_replay_commit_supported(self.attn_config.dtype))

    @override
    def workspace_bytes(self) -> int:
        """Return bytes needed to stage the GDN verify window."""
        if not self.num_draft_layers:
            return 0
        verify_rows = self.attn_config.max_bs * (
            int(self.server_args.speculative_num_draft_tokens) + 1
        )
        # Replay reconstructs the ssm state, so only the conv checkpoint is
        # staged; the backend reads the same decision off the linear component.
        staged = verify_rows * sum(
            field.payload_bytes
            for _, fields in self.groups()
            for field in fields
            if field.field_id.endswith(self._verify_workspace_field_suffixes())
        )
        if self.replay_ssm:
            return staged + self._replay_payload_bytes()
        return staged

    def _verify_workspace_field_suffixes(self) -> tuple[str, ...]:
        """Return cache-field suffixes staged during target verification."""
        return (".conv",) if self.replay_ssm else (".conv", ".ssm")

    def _replay_payload_bytes(self) -> int:
        """Captured verify projections, stacked per GDN layer.

        Mirrors ``_GDNReplayWorkspace``: one key/value/a/b payload row per
        draft position in the model dtype, plus the per-layer fp32
        ``A_log``/``dt_bias`` pair.
        """
        conv_shape, _, ssm_shape, _ = self._state_shapes
        num_v_heads, head_v_dim, _ = ssm_shape
        key_width = (conv_shape[0] - num_v_heads * head_v_dim) // 2
        row_width = key_width + num_v_heads * head_v_dim + 2 * num_v_heads
        num_layers = sum(
            layer_type == LINEAR_ATTENTION for layer_type in self.target_layer_types
        )
        rows = self.attn_config.max_bs * int(
            self.server_args.speculative_num_draft_tokens
        )
        payload = num_layers * rows * row_width * self.attn_config.dtype.itemsize
        parameters = num_layers * 2 * num_v_heads * torch.float32.itemsize
        return payload + parameters

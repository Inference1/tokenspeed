# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Torch reference GDN kernels for Ascend correctness bring-up.

These implementations match the public K-last state-pool contract used by
``gdn_chunk_prefill`` / ``gdn_decode_step``. Prefer the Triton-Ascend path
when it is registered and selected; this module is the Ascend fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchGdnChunkPrefillResult:
    out: torch.Tensor
    final_state: torch.Tensor | None
    h: torch.Tensor | None = None


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    return (
        x_float * torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True).clamp_min(eps))
    ).to(x.dtype)


def _softplus_gate(
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    x = a.float() + dt_bias.float()
    beta_x = softplus_beta * x
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x)),
        x,
    )
    return -torch.exp(A_log.float()) * softplus_x


def gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool = False,
    output_final_state: bool = True,
    output_h: bool = False,
) -> TorchGdnChunkPrefillResult:
    """Varlen GDN chunk prefill (token recurrence). Public state is K-last."""
    if qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim**-0.5

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    beta_f = beta.float()
    initial_state_kv = initial_state.transpose(-2, -1)

    batch, total_tokens, num_q_heads, _ = q.shape
    if batch != 1:
        raise ValueError(f"gdn_chunk_prefill torch path expects batch=1, got {batch}")
    num_v_heads = v.shape[2]
    head_v_dim = v.shape[-1]
    group_size = num_v_heads // num_q_heads

    out = torch.empty(
        (1, total_tokens, num_v_heads, head_v_dim),
        device=q.device,
        dtype=torch.float32,
    )
    h_rows: list[torch.Tensor] | None = [] if output_h else None
    final_states = []
    starts = cu_seqlens[:-1].to(torch.int64).tolist()
    ends = cu_seqlens[1:].to(torch.int64).tolist()

    for seq_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
        state = initial_state_kv[seq_idx].float().clone()
        for token_idx in range(start, end):
            for value_head in range(num_v_heads):
                qk_head = value_head // group_size
                q_t = q_f[0, token_idx, qk_head]
                k_t = k_f[0, token_idx, qk_head]
                v_t = v_f[0, token_idx, value_head]
                state_h = torch.exp(g_f[0, token_idx, value_head]) * state[value_head]
                delta = beta_f[0, token_idx, value_head] * (v_t - k_t @ state_h)
                state_h = state_h + k_t[:, None] * delta[None, :]
                out[0, token_idx, value_head] = scale * (q_t @ state_h)
                state[value_head] = state_h
            if h_rows is not None:
                h_rows.append(state.transpose(-2, -1).clone())
        final_states.append(state)

    final_state_kv = torch.stack(final_states, dim=0).to(initial_state.dtype)
    final_state = final_state_kv.transpose(-2, -1) if output_final_state else None
    h = None
    if h_rows is not None:
        # FLA layout is unused for the torch path; return K-last token states.
        h = torch.stack(h_rows, dim=0)
    return TorchGdnChunkPrefillResult(
        out=out.to(q.dtype),
        final_state=final_state,
        h=h,
    )


def gdn_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    output_state_indices: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Single-step GDN decode with K-last pool updates."""
    return _gdn_decode_update(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        scale=scale,
        output_state_indices=output_state_indices,
        use_qk_l2norm=use_qk_l2norm,
        intermediate_states_buffer=None,
        per_token_output_state_indices=None,
    )


def gdn_decode_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    intermediate_states_buffer: torch.Tensor | None = None,
    per_token_output_state_indices: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-token GDN verify/decode using the same recurrence as decode_step."""
    return _gdn_decode_update(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        scale=scale,
        output_state_indices=output_state_indices,
        use_qk_l2norm=use_qk_l2norm,
        intermediate_states_buffer=intermediate_states_buffer,
        per_token_output_state_indices=per_token_output_state_indices,
    )


def _gdn_decode_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None,
    output_state_indices: torch.Tensor | None,
    use_qk_l2norm: bool,
    intermediate_states_buffer: torch.Tensor | None,
    per_token_output_state_indices: torch.Tensor | None,
) -> torch.Tensor:
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    if scale is None:
        scale = K**-0.5
    group = HV // H

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    a_f = a.float()
    b_f = b.float()
    out = q.new_empty(B, T, HV, V)

    for bi in range(B):
        idx = int(initial_state_indices[bi].item())
        if idx < 0:
            # Match Triton/FlashInfer: skipped slots leave output undefined.
            continue
        # Work in FLA layout [HV, K, V] then write K-last.
        state = initial_state[idx].float().transpose(-2, -1).clone()
        for t in range(T):
            g = _softplus_gate(a_f[bi, t], dt_bias, A_log)
            beta = torch.sigmoid(b_f[bi, t])
            for hv in range(HV):
                qh = hv // group
                q_t = q_f[bi, t, qh]
                k_t = k_f[bi, t, qh]
                if use_qk_l2norm:
                    q_t = q_t / torch.sqrt((q_t * q_t).sum() + 1e-6)
                    k_t = k_t / torch.sqrt((k_t * k_t).sum() + 1e-6)
                st = state[hv] * torch.exp(g[hv])
                delta = beta[hv] * (v_f[bi, t, hv] - k_t @ st)
                st = st + k_t[:, None] * delta[None, :]
                out[bi, t, hv] = (scale * (q_t @ st)).to(out.dtype)
                state[hv] = st

            state_klast = state.transpose(-2, -1)
            if intermediate_states_buffer is not None:
                intermediate_states_buffer[bi, t].copy_(state_klast)
            if per_token_output_state_indices is not None:
                write_idx = int(per_token_output_state_indices[bi, t].item())
                if write_idx >= 0:
                    initial_state[write_idx].copy_(state_klast.to(initial_state.dtype))

        state_klast = state.transpose(-2, -1).to(initial_state.dtype)
        if output_state_indices is not None:
            write_idx = int(output_state_indices[bi].item())
            if write_idx >= 0:
                initial_state[write_idx].copy_(state_klast)
        elif per_token_output_state_indices is None:
            initial_state[idx].copy_(state_klast)

    return out

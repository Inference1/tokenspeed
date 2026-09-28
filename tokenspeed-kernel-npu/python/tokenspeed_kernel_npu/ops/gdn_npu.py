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

"""Ascend-accelerated GDN kernels for TokenSpeed.

Prefers CANN fused ops used by vLLM-Ascend (`npu_chunk_gated_delta_rule`,
`npu_recurrent_gated_delta_rule` / AscendC `_C_ascend`), with a vectorized
Torch path when those ops are missing. Public state layout stays **K-last**
``[N, HV, K, V]`` to match the TokenSpeed contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import torch_npu
except ImportError:  # pragma: no cover
    torch_npu = None  # type: ignore[assignment]


@dataclass(frozen=True)
class NpuGdnChunkPrefillResult:
    out: torch.Tensor
    final_state: torch.Tensor | None
    h: torch.Tensor | None = None


_CHUNK_FUSED_AVAILABLE: bool | None = None
_RECURRENT_OP = None  # callable | False after probe


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


def probe_chunk_fused() -> bool:
    """Whether ``torch_npu.npu_chunk_gated_delta_rule`` works (cached).

    Probe mirrors vLLM-Ascend ``AscendGatedDeltaNetAttention._probe_fused_chunk``
    (Apache-2.0): Dk == Dv == 128 smoke on current NPU.
    """
    global _CHUNK_FUSED_AVAILABLE
    if _CHUNK_FUSED_AVAILABLE is not None:
        return _CHUNK_FUSED_AVAILABLE
    if torch_npu is None or not hasattr(torch_npu, "npu_chunk_gated_delta_rule"):
        _CHUNK_FUSED_AVAILABLE = False
        return False
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        _CHUNK_FUSED_AVAILABLE = False
        return False
    try:
        device = torch.npu.current_device()
        dk = dv = 128
        nk = nv = 1
        seqlen = 64
        q = torch.zeros((seqlen, nk, dk), dtype=torch.bfloat16, device=device)
        k = torch.zeros((seqlen, nk, dk), dtype=torch.bfloat16, device=device)
        v = torch.zeros((seqlen, nv, dv), dtype=torch.bfloat16, device=device)
        beta = torch.full((seqlen, nv), 0.5, dtype=torch.bfloat16, device=device)
        g = torch.full((seqlen, nv), -0.1, dtype=torch.float32, device=device)
        initial_state = torch.zeros(
            (1, nv, dv, dk), dtype=torch.bfloat16, device=device
        )
        actual_seq_lengths = torch.tensor([seqlen], dtype=torch.int32, device=device)
        torch_npu.npu_chunk_gated_delta_rule(
            q,
            k,
            v,
            beta=beta,
            initial_state=initial_state,
            actual_seq_lengths=actual_seq_lengths,
            scale=dk**-0.5,
            g=g,
        )
        torch.npu.synchronize()
        _CHUNK_FUSED_AVAILABLE = True
    except Exception:
        _CHUNK_FUSED_AVAILABLE = False
    return _CHUNK_FUSED_AVAILABLE


def _resolve_recurrent_op():
    global _RECURRENT_OP
    if _RECURRENT_OP is not None:
        return False if _RECURRENT_OP is False else _RECURRENT_OP
    candidates = []
    if torch_npu is not None and hasattr(torch_npu, "npu_recurrent_gated_delta_rule"):
        candidates.append(torch_npu.npu_recurrent_gated_delta_rule)
    try:
        op = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
        candidates.append(op)
    except (AttributeError, RuntimeError):
        pass
    _RECURRENT_OP = candidates[0] if candidates else False
    return False if _RECURRENT_OP is False else _RECURRENT_OP


def _chunk_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool,
) -> NpuGdnChunkPrefillResult:
    """Call CANN fused chunk GDN. initial_state / final_state are K-last."""
    assert torch_npu is not None
    if qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    # TokenSpeed K-last [N,HV,K,V] -> fused [N,HV,V,K]
    init_vk = initial_state.transpose(-2, -1).contiguous().to(torch.bfloat16)
    q_t = q.squeeze(0).contiguous()
    k_t = k.squeeze(0).contiguous()
    v_t = v.squeeze(0).contiguous()
    g_t = g.squeeze(0).to(torch.float32).contiguous()
    beta_t = beta.squeeze(0).to(v.dtype).contiguous()
    actual_seq_lengths = torch.diff(cu_seqlens).to(torch.int32)
    out_t, final_vk = torch_npu.npu_chunk_gated_delta_rule(
        q_t,
        k_t,
        v_t,
        beta=beta_t,
        initial_state=init_vk,
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g_t,
    )
    final_state = final_vk.transpose(-2, -1).contiguous().to(initial_state.dtype)
    return NpuGdnChunkPrefillResult(
        out=out_t.unsqueeze(0).to(q.dtype),
        final_state=final_state,
        h=None,
    )


def _vectorized_token_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    state: torch.Tensor,
    *,
    scale: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One token, all V heads. ``state`` is FLA ``[HV, K, V]``."""
    hv = v_t.shape[0]
    q_h = q_t.shape[0]
    if group_size != hv // q_h:
        group_size = hv // q_h
    q_exp = q_t.repeat_interleave(group_size, dim=0)
    k_exp = k_t.repeat_interleave(group_size, dim=0)
    # state: [HV, K, V]; decay on K rows via g on HV
    decay = torch.exp(g_t).to(state.dtype)
    state_h = state * decay[:, None, None]
    # k @ state -> [HV, V]
    kv = torch.einsum("hk,hkv->hv", k_exp, state_h)
    delta = beta_t[:, None] * (v_t - kv)
    state_h = state_h + torch.einsum("hk,hv->hkv", k_exp, delta)
    out = scale * torch.einsum("hk,hkv->hv", q_exp, state_h)
    return out, state_h


def _vectorized_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool,
    output_final_state: bool,
    output_h: bool,
) -> NpuGdnChunkPrefillResult:
    if qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    batch, total_tokens, num_q_heads, head_dim = q.shape
    if batch != 1:
        raise ValueError(f"gdn_chunk_prefill expects batch=1, got {batch}")
    num_v_heads = v.shape[2]
    head_v_dim = v.shape[-1]
    group_size = num_v_heads // num_q_heads
    q_f = q[0].float()
    k_f = k[0].float()
    v_f = v[0].float()
    g_f = g[0].float()
    beta_f = beta[0].float()
    out = q.new_empty(1, total_tokens, num_v_heads, head_v_dim)
    h_rows: list[torch.Tensor] | None = [] if output_h else None
    final_states = []
    starts = cu_seqlens[:-1].to(torch.int64).tolist()
    ends = cu_seqlens[1:].to(torch.int64).tolist()
    for seq_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
        # K-last -> FLA [HV, K, V]
        state = initial_state[seq_idx].float().transpose(-2, -1).clone()
        for token_idx in range(start, end):
            o_t, state = _vectorized_token_step(
                q_f[token_idx],
                k_f[token_idx],
                v_f[token_idx],
                g_f[token_idx],
                beta_f[token_idx],
                state,
                scale=scale,
                group_size=group_size,
            )
            out[0, token_idx] = o_t.to(out.dtype)
            if h_rows is not None:
                h_rows.append(state.transpose(-2, -1).clone())
        final_states.append(state.transpose(-2, -1))
    final_state = (
        torch.stack(final_states, dim=0).to(initial_state.dtype)
        if output_final_state
        else None
    )
    h = torch.stack(h_rows, dim=0) if h_rows is not None else None
    return NpuGdnChunkPrefillResult(out=out, final_state=final_state, h=h)


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
) -> NpuGdnChunkPrefillResult:
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim**-0.5
    # Fused CANN path does not emit intermediate h; fall back if requested.
    if (
        not output_h
        and output_final_state
        and head_dim == 128
        and v.shape[-1] == 128
        and probe_chunk_fused()
    ):
        try:
            return _chunk_fused(
                q,
                k,
                v,
                g,
                beta,
                scale=float(scale),
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
                qk_l2norm=qk_l2norm,
            )
        except Exception:
            pass
    return _vectorized_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=float(scale),
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        qk_l2norm=qk_l2norm,
        output_final_state=output_final_state,
        output_h=output_h,
    )


def _vectorized_decode_update(
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
    B, T, H, Kdim = q.shape
    HV = v.shape[2]
    Vdim = v.shape[3]
    if scale is None:
        scale = Kdim**-0.5
    group = HV // H
    out = q.new_empty(B, T, HV, Vdim)
    for bi in range(B):
        idx = int(initial_state_indices[bi].item())
        if idx < 0:
            continue
        state = initial_state[idx].float().transpose(-2, -1).clone()
        for t in range(T):
            g = _softplus_gate(a[bi, t], dt_bias, A_log)
            beta = torch.sigmoid(b[bi, t].float())
            q_t = q[bi, t].float()
            k_t = k[bi, t].float()
            if use_qk_l2norm:
                q_t = _l2norm(q_t.unsqueeze(0)).squeeze(0)
                k_t = _l2norm(k_t.unsqueeze(0)).squeeze(0)
            o_t, state = _vectorized_token_step(
                q_t,
                k_t,
                v[bi, t].float(),
                g,
                beta,
                state,
                scale=float(scale),
                group_size=group,
            )
            out[bi, t] = o_t.to(out.dtype)
            state_klast = state.transpose(-2, -1)
            if intermediate_states_buffer is not None:
                intermediate_states_buffer[bi, t].copy_(state_klast)
            if per_token_output_state_indices is not None:
                write_idx = int(per_token_output_state_indices[bi, t].item())
                if write_idx >= 0:
                    initial_state[write_idx].copy_(
                        state_klast.to(initial_state.dtype)
                    )
        state_klast = state.transpose(-2, -1).to(initial_state.dtype)
        if output_state_indices is not None:
            write_idx = int(output_state_indices[bi].item())
            if write_idx >= 0:
                initial_state[write_idx].copy_(state_klast)
        elif per_token_output_state_indices is None:
            initial_state[idx].copy_(state_klast)
    return out


def _try_recurrent_decode(
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
    use_qk_l2norm: bool,
) -> torch.Tensor | None:
    """Best-effort CANN recurrent decode for T==1 dense batches.

    Returns None if the custom OPP is unavailable or the call fails so the
    caller can fall back to the vectorized Torch path.
    """
    op = _resolve_recurrent_op()
    if op is False or q.shape[1] != 1:
        return None
    B, _, H, Kdim = q.shape
    HV = v.shape[2]
    if scale is None:
        scale = Kdim**-0.5
    try:
        # Build TND-like packed tensors [B, H/HV, D] after optional L2.
        q_u = q[:, 0]
        k_u = k[:, 0]
        if use_qk_l2norm:
            q_u = _l2norm(q_u)
            k_u = _l2norm(k_u)
        g = _softplus_gate(a[:, 0], dt_bias, A_log)  # [B, HV]
        beta = torch.sigmoid(b[:, 0].float()).to(v.dtype)
        # CANN recurrent ops typically keep SSM as [pool, HV, V, K] (Dv,Dk).
        # Work on a transposed view copy for the active indices only when needed.
        # Many builds mutate ``state`` in-place via indices — pass K-last pool
        # transposed workspace.
        state_vk = initial_state.transpose(-2, -1).contiguous()
        actual_seq_lengths = torch.ones(B, dtype=torch.int32, device=q.device)
        # Pack as [T=B?, ...] — vLLM uses squeeze(0) on batch=1 varlen. For
        # TokenSpeed decode, batch is request batch; flatten to TND over batch.
        query = q_u  # [B, H, K]
        key = k_u
        value = v[:, 0]
        out = op(
            query=query,
            key=key,
            value=value,
            g=g.to(torch.float32),
            beta=beta,
            state=state_vk,
            scale=float(scale),
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=initial_state_indices.to(torch.int32),
        )
        # Write VK state back to K-last pool.
        initial_state.copy_(state_vk.transpose(-2, -1).to(initial_state.dtype))
        if out.ndim == 3:
            out = out.unsqueeze(1)
        return out.to(q.dtype)
    except Exception:
        return None


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
    if output_state_indices is None:
        fused = _try_recurrent_decode(
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
            use_qk_l2norm=use_qk_l2norm,
        )
        if fused is not None:
            return fused
    return _vectorized_decode_update(
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
    return _vectorized_decode_update(
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


# Re-export softplus for tests / parity with torch path.
softplus_gate = _softplus_gate

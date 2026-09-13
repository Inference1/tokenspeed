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

"""Ascend multi-head attention kernels."""

from __future__ import annotations

import math

import torch
import torch_npu

# Ascend FusedInferAttentionScore TND layout only accepts these head dims
# (or Q/K=192 with V=128). Qwen3.5/3.8 full-attn uses head_dim=256.
_TND_SUPPORTED_HEAD_DIMS = frozenset({64, 128, 192})

_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def _causal_mask(device: torch.device) -> torch.Tensor:
    mask = _CAUSAL_MASKS.get(device)
    if mask is None:
        mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1
        )
        _CAUSAL_MASKS[device] = mask
    return mask


def _scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    return softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def _check_options(
    *,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    return_lse: bool,
) -> None:
    if window_left >= 0:
        raise NotImplementedError("Ascend MHA does not support sliding windows")
    if logit_cap:
        raise NotImplementedError("Ascend MHA does not support logit caps")
    if sinks is not None:
        raise NotImplementedError("Ascend MHA does not support attention sinks")
    if return_lse:
        raise NotImplementedError("Ascend MHA does not return LSE")


def _needs_eager(q: torch.Tensor) -> bool:
    return int(q.shape[-1]) not in _TND_SUPPORTED_HEAD_DIMS


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def _eager_seq_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Single-sequence attention. Tensors are ``[L, H, D]``."""
    n_rep = q.shape[1] // k.shape[1]
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)
    # [H, L, D]
    qh = q.transpose(0, 1)
    kh = k.transpose(0, 1)
    vh = v.transpose(0, 1)
    scores = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) * scale
    if causal:
        q_len, k_len = qh.shape[1], kh.shape[1]
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device)
        if q_len == k_len:
            mask = torch.triu(mask, diagonal=1)
        else:
            # Extend/decode: queries attend to full prefix + their own positions.
            offset = k_len - q_len
            mask = torch.triu(mask, diagonal=1 + offset)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
    out = torch.matmul(probs, vh.to(dtype=probs.dtype)).to(dtype=q.dtype)
    return out.transpose(0, 1).contiguous()


def _eager_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_cpu: list[int],
    scale: float,
) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    for i in range(len(cu_seqlens_cpu) - 1):
        s, e = int(cu_seqlens_cpu[i]), int(cu_seqlens_cpu[i + 1])
        if e <= s:
            continue
        outs.append(
            _eager_seq_attn(q[s:e], k[s:e], v[s:e], scale=scale, causal=True)
        )
    if not outs:
        return torch.empty_like(q)
    return torch.cat(outs, dim=0)


def _gather_paged_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table_row: torch.Tensor,
    seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ``[seqlen, H_kv, D]`` from paged cache for one request."""
    block_size = int(k_cache.shape[1])
    num_pages = (int(seqlen) + block_size - 1) // block_size
    pages = page_table_row[:num_pages].to(dtype=torch.long)
    k = k_cache[pages].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:seqlen]
    v = v_cache[pages].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:seqlen]
    return k, v


def mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run causal variable-length MHA over uncached K/V."""
    del cu_seqlens, max_seqlen
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    scale = _scale(q, softmax_scale)
    if _needs_eager(q):
        return _eager_mha_prefill(q, k, v, cu_seqlens_cpu, scale)

    output, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        atten_mask=_causal_mask(q.device),
        actual_seq_lengths=cu_seqlens_cpu[1:],
        actual_seq_lengths_kv=cu_seqlens_cpu[1:],
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=scale,
        input_layout="TND",
        sparse_mode=2,
    )
    return output


def mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run variable-length MHA over a paged K/V cache."""
    del cu_seqlens_kv, max_seqlen_q, max_seqlen_k, enable_pdl
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    scale = _scale(q, softmax_scale)
    if _needs_eager(q):
        cu_q = cu_seqlens_q.tolist()
        cache_lens = cache_seqlens.tolist()
        outs: list[torch.Tensor] = []
        for i in range(len(cu_q) - 1):
            qs, qe = int(cu_q[i]), int(cu_q[i + 1])
            kv_len = int(cache_lens[i])
            qi = q[qs:qe]
            ki, vi = _gather_paged_kv(k_cache, v_cache, page_table[i], kv_len)
            outs.append(
                _eager_seq_attn(qi, ki, vi, scale=scale, causal=is_causal)
            )
        return torch.cat(outs, dim=0) if outs else torch.empty_like(q)

    output, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k_cache.flatten(2),
        v_cache.flatten(2),
        atten_mask=_causal_mask(q.device) if is_causal else None,
        actual_seq_lengths=cu_seqlens_q[1:],
        actual_seq_lengths_kv=cache_seqlens,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=scale,
        input_layout="TND",
        sparse_mode=3 if is_causal else 0,
        block_size=k_cache.shape[1],
    )
    return output


def mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run fixed-shape, graph-capturable decode over a paged K/V cache."""
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if max_seqlen_q != 1:
        raise NotImplementedError("Ascend MHA decode supports one query per request")
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    del max_seqlen_k, enable_pdl
    scale = _scale(q, softmax_scale)
    batch_size = cache_seqlens.shape[0]

    if _needs_eager(q):
        # q: [B, H, D] or [B, 1, H, D] depending on caller; normalize to [1,H,D] per row.
        if q.ndim == 3:
            q_rows = q
        else:
            q_rows = q.reshape(batch_size, q.shape[-2], q.shape[-1])
        outs: list[torch.Tensor] = []
        cache_lens = cache_seqlens.tolist()
        for i in range(batch_size):
            qi = q_rows[i : i + 1]
            ki, vi = _gather_paged_kv(
                k_cache, v_cache, page_table[i], int(cache_lens[i])
            )
            outs.append(_eager_seq_attn(qi, ki, vi, scale=scale, causal=False))
        return torch.cat(outs, dim=0).reshape_as(q)

    actual_seq_lengths_kv = (
        [1] * batch_size if torch.npu.is_current_stream_capturing() else cache_seqlens
    )
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q.reshape(batch_size, 1, -1),
        k_cache.flatten(2),
        v_cache.flatten(2),
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=scale,
        input_layout="BSH",
        block_size=k_cache.shape[1],
    )
    return output.reshape_as(q)


__all__ = [
    "mha_decode_with_kvcache",
    "mha_extend_with_kvcache",
    "mha_prefill",
]

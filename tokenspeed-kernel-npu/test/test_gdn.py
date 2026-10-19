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

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops import gdn as torch_gdn  # noqa: E402
from tokenspeed_kernel_npu.ops.gdn import TorchGdnChunkPrefillResult  # noqa: E402

# Optional: full registry path when tokenspeed_kernel is importable on NPU.
try:
    from tokenspeed_kernel.ops.attention import (  # noqa: E402
        GdnChunkPrefillResult,
        gdn_chunk_prefill,
        gdn_decode_step,
    )
except Exception:  # pragma: no cover
    GdnChunkPrefillResult = TorchGdnChunkPrefillResult  # type: ignore[misc,assignment]
    gdn_chunk_prefill = None  # type: ignore[assignment]
    gdn_decode_step = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend GDN tests require an NPU"
)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    return (
        x_float * torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True).clamp_min(eps))
    ).to(x.dtype)


@pytest.mark.parametrize("solution", ["torch", "triton"])
def test_gdn_chunk_prefill_matches_torch_oracle(solution: str) -> None:
    if gdn_chunk_prefill is None:
        pytest.skip("tokenspeed_kernel registry unavailable")
    torch.manual_seed(1234)
    device = "npu"
    seq_len = 32
    num_q_heads = 4
    num_v_heads = 8
    head_dim = 64
    dtype = torch.bfloat16
    q = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_v_heads, head_dim, device=device, dtype=dtype) * 0.5
    g = F.logsigmoid(
        torch.randn(1, seq_len, num_v_heads, device=device, dtype=torch.float32)
    )
    beta = torch.rand(1, seq_len, num_v_heads, device=device, dtype=dtype).sigmoid()
    initial_state = (
        torch.randn(1, num_v_heads, head_dim, head_dim, device=device, dtype=dtype)
        * 0.01
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    scale = head_dim**-0.5

    try:
        result = gdn_chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state.clone(),
            cu_seqlens=cu_seqlens,
            qk_l2norm=True,
            output_final_state=True,
            solution=solution,
        )
    except Exception as exc:  # pragma: no cover - solution may be unavailable
        if solution == "triton":
            pytest.skip(f"triton GDN unavailable on this Ascend stack: {exc}")
        raise

    assert isinstance(result, GdnChunkPrefillResult)
    ref = torch_gdn.gdn_chunk_prefill(
        _l2norm(q),
        _l2norm(k),
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=False,
        output_final_state=True,
    )
    torch.testing.assert_close(
        result.out.float(), ref.out.float(), rtol=3e-2, atol=3e-2
    )
    torch.testing.assert_close(
        result.final_state.float(), ref.final_state.float(), rtol=3e-2, atol=3e-2
    )


def test_gdn_decode_step_torch_updates_pool() -> None:
    if gdn_decode_step is None:
        pytest.skip("tokenspeed_kernel registry unavailable")
    torch.manual_seed(7)
    device = "npu"
    dtype = torch.bfloat16
    batch = 2
    num_q_heads = 2
    num_v_heads = 4
    head_dim = 64
    pool_size = 8
    q = torch.randn(batch, 1, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, 1, num_q_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, 1, num_v_heads, head_dim, device=device, dtype=dtype) * 0.5
    a = torch.randn(batch, 1, num_v_heads, device=device, dtype=torch.float32)
    b = torch.randn(batch, 1, num_v_heads, device=device, dtype=torch.float32)
    A_log = torch.randn(num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(num_v_heads, device=device, dtype=torch.float32)
    pool = (
        torch.randn(
            pool_size, num_v_heads, head_dim, head_dim, device=device, dtype=dtype
        )
        * 0.02
    )
    read_idx = torch.tensor([1, 3], device=device, dtype=torch.int32)
    scale = head_dim**-0.5

    pool_before = pool.clone()
    out = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool,
        initial_state_indices=read_idx,
        scale=scale,
        use_qk_l2norm=True,
        solution="torch",
    )
    assert out.shape == v.shape
    assert not torch.equal(pool[read_idx], pool_before[read_idx])
    untouched = torch.tensor([0, 2, 4, 5, 6, 7], device=device)
    torch.testing.assert_close(pool[untouched], pool_before[untouched])

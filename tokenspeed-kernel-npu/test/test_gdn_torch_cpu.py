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

"""CPU-side oracle checks for Ascend Torch GDN (no NPU required)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tokenspeed_kernel_npu.ops import gdn as torch_gdn


def test_torch_gdn_chunk_prefill_cpu_smoke() -> None:
    torch.manual_seed(0)
    seq_len = 16
    num_q_heads = 2
    num_v_heads = 4
    head_dim = 32
    dtype = torch.bfloat16
    q = torch.randn(1, seq_len, num_q_heads, head_dim, dtype=dtype)
    k = torch.randn(1, seq_len, num_q_heads, head_dim, dtype=dtype)
    v = torch.randn(1, seq_len, num_v_heads, head_dim, dtype=dtype) * 0.5
    g = F.logsigmoid(torch.randn(1, seq_len, num_v_heads, dtype=torch.float32))
    beta = torch.rand(1, seq_len, num_v_heads, dtype=dtype).sigmoid()
    initial_state = torch.zeros(1, num_v_heads, head_dim, head_dim, dtype=dtype)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
    result = torch_gdn.gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=head_dim**-0.5,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
    )
    assert result.out.shape == (1, seq_len, num_v_heads, head_dim)
    assert result.final_state is not None
    assert result.final_state.shape == initial_state.shape
    assert torch.isfinite(result.out.float()).all()


def test_torch_gdn_decode_step_cpu_smoke() -> None:
    torch.manual_seed(1)
    batch = 2
    num_q_heads = 2
    num_v_heads = 4
    head_dim = 32
    dtype = torch.bfloat16
    q = torch.randn(batch, 1, num_q_heads, head_dim, dtype=dtype)
    k = torch.randn(batch, 1, num_q_heads, head_dim, dtype=dtype)
    v = torch.randn(batch, 1, num_v_heads, head_dim, dtype=dtype) * 0.5
    a = torch.randn(batch, 1, num_v_heads, dtype=torch.float32)
    b = torch.randn(batch, 1, num_v_heads, dtype=torch.float32)
    A_log = torch.randn(num_v_heads, dtype=torch.float32)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32)
    pool = torch.zeros(4, num_v_heads, head_dim, head_dim, dtype=dtype)
    idx = torch.tensor([0, 2], dtype=torch.int32)
    out = torch_gdn.gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool,
        initial_state_indices=idx,
        scale=head_dim**-0.5,
        use_qk_l2norm=True,
    )
    assert out.shape == v.shape
    assert torch.isfinite(out.float()).all()
    assert not torch.equal(pool[0], torch.zeros_like(pool[0]))

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

"""CPU-side parity: vectorized Ascend GDN vs token-loop Torch oracle."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tokenspeed_kernel_npu.ops import gdn as torch_gdn
from tokenspeed_kernel_npu.ops import gdn_npu


def test_vectorized_chunk_prefill_matches_torch_oracle() -> None:
    torch.manual_seed(0)
    seq_len = 16
    num_q_heads = 2
    num_v_heads = 4
    head_dim = 32
    dtype = torch.float32
    q = torch.randn(1, seq_len, num_q_heads, head_dim, dtype=dtype)
    k = torch.randn(1, seq_len, num_q_heads, head_dim, dtype=dtype)
    v = torch.randn(1, seq_len, num_v_heads, head_dim, dtype=dtype) * 0.5
    g = F.logsigmoid(torch.randn(1, seq_len, num_v_heads, dtype=dtype))
    beta = torch.rand(1, seq_len, num_v_heads, dtype=dtype).sigmoid()
    initial_state = (
        torch.randn(1, num_v_heads, head_dim, head_dim, dtype=dtype) * 0.01
    )
    cu = torch.tensor([0, seq_len], dtype=torch.int32)
    scale = head_dim**-0.5

    # Force vectorized path (fused CANN only for D=128 on NPU).
    got = gdn_npu._vectorized_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu,
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
    )
    ref = torch_gdn.gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu,
        qk_l2norm=True,
        output_final_state=True,
    )
    torch.testing.assert_close(got.out, ref.out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        got.final_state, ref.final_state, rtol=1e-4, atol=1e-4
    )


def test_vectorized_decode_step_matches_torch_oracle() -> None:
    torch.manual_seed(1)
    batch = 2
    num_q_heads = 2
    num_v_heads = 4
    head_dim = 32
    pool_size = 4
    dtype = torch.float32
    q = torch.randn(batch, 1, num_q_heads, head_dim, dtype=dtype)
    k = torch.randn(batch, 1, num_q_heads, head_dim, dtype=dtype)
    v = torch.randn(batch, 1, num_v_heads, head_dim, dtype=dtype) * 0.5
    a = torch.randn(batch, 1, num_v_heads, dtype=dtype)
    b = torch.randn(batch, 1, num_v_heads, dtype=dtype)
    A_log = torch.randn(num_v_heads, dtype=dtype)
    dt_bias = torch.randn(num_v_heads, dtype=dtype)
    pool = torch.randn(pool_size, num_v_heads, head_dim, head_dim, dtype=dtype) * 0.02
    idx = torch.tensor([0, 2], dtype=torch.int32)
    scale = head_dim**-0.5

    pool_a = pool.clone()
    pool_b = pool.clone()
    out_a = gdn_npu._vectorized_decode_update(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_a,
        initial_state_indices=idx,
        scale=scale,
        output_state_indices=None,
        use_qk_l2norm=True,
        intermediate_states_buffer=None,
        per_token_output_state_indices=None,
    )
    out_b = torch_gdn.gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_b,
        initial_state_indices=idx,
        scale=scale,
        use_qk_l2norm=True,
    )
    torch.testing.assert_close(out_a, out_b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pool_a, pool_b, rtol=1e-4, atol=1e-4)

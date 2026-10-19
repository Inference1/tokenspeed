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

"""Ascend attention kernel registrations."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel.ops.attention import GdnChunkPrefillResult
    from tokenspeed_kernel_npu.ops import gdn as _gdn
    from tokenspeed_kernel_npu.ops.mha import (
        mha_decode_with_kvcache as _mha_decode_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mha import (
        mha_extend_with_kvcache as _mha_extend_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mha import mha_prefill as _mha_prefill

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    _DTYPES = {torch.float16, torch.bfloat16}
    _OPTIONS = {
        "sliding_window": frozenset({False}),
        "support_sinks": frozenset({False}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
    }

    @register_kernel(
        "attention",
        "mha_prefill",
        name="ascend_mha_prefill",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits=_OPTIONS,
        tags={"portability"},
    )
    def mha_prefill(**kwargs):
        return _mha_prefill(**kwargs)

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="ascend_mha_extend_with_kvcache",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k_cache", "v_cache"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            **_OPTIONS,
            "page_size": frozenset({64, 128}),
            "is_causal": frozenset({False, True}),
        },
        tags={"portability"},
    )
    def mha_extend_with_kvcache(**kwargs):
        return _mha_extend_with_kvcache(**kwargs)

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="ascend_mha_decode_with_kvcache",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k_cache", "v_cache"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            **_OPTIONS,
            "page_size": frozenset({64, 128}),
            "q_len": frozenset({1}),
        },
        tags={"portability"},
    )
    def mha_decode_with_kvcache(**kwargs):
        return _mha_decode_with_kvcache(**kwargs)

    # Torch GDN fallbacks (priority below Triton PORTABLE). Selection prefers
    # triton_gdn_* once Ascend is included in that capability set.
    _GDN_TRAITS = {
        "qk_l2norm": frozenset({False, True}),
        "output_h": frozenset({False, True}),
    }

    @register_kernel(
        "attention",
        "gdn_chunk_prefill",
        name="ascend_torch_gdn_chunk_prefill",
        solution="torch",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=1,
        traits=_GDN_TRAITS,
        tags={"portability", "ascend-fallback"},
    )
    def gdn_chunk_prefill(**kwargs):
        result = _gdn.gdn_chunk_prefill(**kwargs)
        return GdnChunkPrefillResult(
            out=result.out,
            final_state=result.final_state,
            h=result.h,
        )

    @register_kernel(
        "attention",
        "gdn_decode_step",
        name="ascend_torch_gdn_decode_step",
        solution="torch",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=1,
        tags={"portability", "ascend-fallback"},
    )
    def gdn_decode_step(**kwargs):
        return _gdn.gdn_decode_step(**kwargs)

    @register_kernel(
        "attention",
        "gdn_decode_mtp",
        name="ascend_torch_gdn_decode_mtp",
        solution="torch",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=1,
        tags={"portability", "speculative-decoding", "ascend-fallback"},
    )
    def gdn_decode_mtp(**kwargs):
        return _gdn.gdn_decode_mtp(**kwargs)


__all__ = [
    "gdn_chunk_prefill",
    "gdn_decode_mtp",
    "gdn_decode_step",
    "mha_decode_with_kvcache",
    "mha_extend_with_kvcache",
    "mha_prefill",
]

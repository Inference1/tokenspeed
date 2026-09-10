# Qwen3.8-27B Ascend gap probe (Phase 0)

## Baseline

- Qwen3-0.6B Ascend serve works with `tokenspeed-kernel-npu` MHA / RMSNorm / RoPE.
- Qwen3.8-27B HF config: `model_type=qwen3_5`, 64 layers, `full_attention_interval=4`
  (48× `linear_attention` GDN + 16× gated `full_attention`), text hidden 5120,
  GDN V/QK heads 48/16 @ 128, full attn Q/KV 24/4 @ 256, `attn_output_gate=true`.

## Registry gap (before this branch)

| Op | CUDA | Ascend (pre) | Ascend (this branch) |
|----|------|--------------|----------------------|
| `mha_*` | many | `torch_npu` | unchanged |
| `gdn_chunk_prefill` | Triton + FlashInfer | **missing** (Triton vendors nvidia/amd only) | Triton vendors include `ascend` + Torch fallback |
| `gdn_decode_step` / `gdn_decode_mtp` | Triton + FlashInfer | **missing** | same as above |
| `sigmoid_mul` (attn gate) | Triton | would fail / unvalidated | eager PyTorch on NPU |
| causal_conv1d / fused_gdn_gating | Triton + PDL gated | PDL off on Ascend | reuse with `pdl_enabled()==False` |

## Expected first serve failure (pre-fix)

`ModuleNotFound` / kernel selection error when hybrid GDN calls
`gdn_chunk_prefill` / `gdn_decode_step` because no Ascend-capable solution was
registered.

## Adaptation strategy

1. Advertise Ascend on portable Triton GDN registrations.
2. Register Torch recurrence fallbacks in `tokenspeed_kernel.ops.attention.ascend`.
3. Keep full-attn on Ascend MHA via `--attention-backend mha` (hybrid wrapper).
4. Eager `sigmoid_mul` on NPU for gated full-attention.
5. Document short-context TP=4/8 BF16 text smoke recipe; defer VLM/MTP/FP8.

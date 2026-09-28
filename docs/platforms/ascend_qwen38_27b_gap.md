# Qwen3.8-27B Ascend gap probe

## Baseline

- Qwen3-0.6B Ascend serve works with `tokenspeed-kernel-npu` MHA / RMSNorm / RoPE.
- Qwen3.8-27B HF config: `model_type=qwen3_5`, 64 layers, `full_attention_interval=4`
  (48× `linear_attention` GDN + 16× gated `full_attention`), text hidden 5120,
  GDN V/QK heads 48/16 @ 128, full attn Q/KV 24/4 @ 256, `attn_output_gate=true`.

## Registry / kernel status

| Op | CUDA | Ascend (bring-up) | Ascend (perf path, landed) |
|----|------|-------------------|----------------------------|
| `mha_*` | many | FIA TND (D∈{64,128,192}); D=256 was eager | **BNSD** `npu_fused_infer_attention_score` for D=256 (probe+cache); else eager |
| `gdn_chunk_prefill` | Triton + FlashInfer | Torch token recurrence (`solution=torch`) | **`solution=torch_npu`** (`ascend_npu_gdn_*`, Priority PERFORMANT+2): CANN `npu_chunk_gated_delta_rule` when Dk=Dv=128 probe OK; else vectorized Torch |
| `gdn_decode_step` / `gdn_decode_mtp` | Triton + FlashInfer | Torch token/head loops | Prefer `npu_recurrent_gated_delta_rule` / `_C_ascend`; else vectorized Torch; Torch PORTABLE fallback retained |
| `sigmoid_mul` (attn gate) | Triton | eager PyTorch on NPU | unchanged |
| causal_conv1d / fused_gdn_gating | Triton + PDL | PDL off; portable path | Future: AscendC `causal_conv1d` / `fused_gdn_gating` from vLLM-Ascend |

## vLLM-Ascend reference map (Apache-2.0)

Source: [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) `vllm_ascend/ops/gdn.py`.

| TokenSpeed contract | vLLM-Ascend / CANN |
|---------------------|--------------------|
| Prefill GDN | `torch_npu.npu_chunk_gated_delta_rule` (TND; state `[N,Nv,Dv,Dk]`; probe+cache availability; Dk=Dv=128 smoke) |
| Prefill fallback | Vectorized Torch recurrence in `tokenspeed_kernel_npu.ops.gdn_npu` (not CUDA Triton) |
| Decode / MTP GDN | `torch.ops._C_ascend.npu_recurrent_gated_delta_rule` (custom OPP via `ASCEND_CUSTOM_OPP_PATH`) or older `torch_npu.npu_recurrent_gated_delta_rule` |
| State layout | vLLM SSM often `[N,Nv,Dv,Dk]`; TokenSpeed public pool is **K-last** `[N,HV,K,V]` → transpose(-2,-1) at boundaries |
| Full attn D=256 | FIA **TND rejects D=256** on many CANN builds; TokenSpeed tries **BNSD** FIA per sequence then eager |

License note: adaptations credit Apache-2.0 vLLM-Ascend / Huawei where APIs are mirrored; TokenSpeed keeps its own registry wrappers and Torch oracles. Files: `ops/gdn_npu.py`, `ops/mha.py` (BNSD), registry in `tokenspeed_kernel/ops/attention/ascend.py`.

## Adaptation strategy (perf)

1. ~~Register Ascend PERFORMANT GDN kernels~~ — done (`ascend_npu_gdn_*` + Torch PORTABLE).
2. ~~For full-attn `head_dim=256`, attempt BNSD FIA~~ — done in `mha.py`.
3. Keep default `--enforce-eager`; optional `ASCEND_ALLOW_GRAPH=1` on a **non-31891** port for ACL graph A/B (`scripts/ascend_qwen38_smoke.sh` + `bench_http`).
4. Do not replace the TokenSpeed scheduler with vLLM; kernel-only port.

## Measured bring-up (correctness-first, eager)

- Chat verify PASS; custom jsonl 7/7.
- EvalScope AIME-2025 limit5 / `max_tokens=2048`: Accuracy 60%.
- EvalScope GPQA Diamond limit10 / `max_tokens=2048`: Accuracy 60%.
- Decode throughput under Torch GDN + eager MHA-256: ~1.3–1.7 tok/s (4× NPU).
- After this port: re-measure with `PORT=31901 bash scripts/ascend_qwen38_bench_http.sh` (eager baseline) vs `ASCEND_ALLOW_GRAPH=1 PORT=31902` when free cards are available; do not interrupt AIME on 31891.

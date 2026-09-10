# Sync local Qwen3.8 Ascend patches onto the lab checkout
#
# From Windows (PowerShell), with SSH password auth interactive:
#   scp -r d:\myMission\tokenspeed\tokenspeed-kernel-npu\python\tokenspeed_kernel_npu root@178.136.2.2:/tmp/ts_npu_pkg
# Then inside the container, copy over the matching paths under
# /home/tokenspeed_ws/tokenspeed/
#
# Or on the Ascend host after pushing this branch:
#   cd /home/tokenspeed_ws/tokenspeed
#   git fetch && git checkout feat/npu-qwen38-27b-gdn
#   bash scripts/ascend_qwen38_smoke.sh

Changed paths in this branch:
- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/gated_delta_rule.py
- tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/ascend.py
- tokenspeed-kernel/python/tokenspeed_kernel/ops/activation/triton.py
- tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/_triton.py
- tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/gdn.py
- tokenspeed-kernel-npu/test/test_gdn.py
- tokenspeed-kernel-npu/test/test_gdn_torch_cpu.py
- tokenspeed-kernel-npu/README.md
- docs/recipes/models.md
- docs/platforms/ascend_qwen38_27b_gap.md
- scripts/ascend_qwen38_smoke.sh

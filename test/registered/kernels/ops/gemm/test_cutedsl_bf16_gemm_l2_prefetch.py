"""The TGV GEMM's weight L2 prefetch only moves bytes into L2.

With PDL and a K-major weight longer than the smem ring, warp 3 prefetches the
remaining weight k-tiles into L2 while DMA_B waits on the predecessor. The
output must be bitwise identical to the same tactic without the prefetch.
"""

import pytest
import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.kernels.ops.gemm import cutedsl_bf16_gemm as tgv  # noqa: E402

# (n, k): K spans several smem rings; the last one fits the ring (no prefetch).
SHAPES = [(8704, 5120), (5120, 4352), (6144, 2048), (2048, 256)]


@pytest.mark.parametrize("m", [1, 7, 16])
@pytest.mark.parametrize("n,k", SHAPES)
def test_l2_prefetch_is_bitwise_neutral(m, n, k):
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")
    torch.manual_seed(m * 7 + n + k)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    tactic = tgv._pick_tactic(m, n, k)
    cta_m, _, num_ab_stage, _ = tgv._resolve_tactic(tactic)

    outs = {}
    for l2_prefetch in (False, True):
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        tgv._run_tgv(x, w.t(), None, out, True, tactic, l2_prefetch=l2_prefetch)
        outs[l2_prefetch] = out
    torch.cuda.synchronize()

    assert torch.equal(outs[True], outs[False])
    ref = (x.float() @ w.float().T).bfloat16()
    torch.testing.assert_close(outs[True], ref, rtol=2e-2, atol=2.5)

    expect = tgv._l2_prefetch_plan(
        budget=tgv._l2_prefetch_budget(x.device.index),
        pdl=True,
        k_major=True,
        k=k,
        cta_m=cta_m,
        num_ab_stage=num_ab_stage,
    )
    compiled = {key[-2:] for key in tgv._TGV_CUTE_EXT_COMPILE_CACHE}
    assert expect in compiled

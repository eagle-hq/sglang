"""Tests for the CuTe DSL TGV BF16 GEMM kernel."""

import sys

import pytest
import torch

from sglang.kernels.jit.utils import (
    get_ci_test_range,
    get_jit_cuda_arch,
    is_hip_runtime,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.kernels.ops.gemm import cutedsl_bf16_gemm as tgv  # noqa: E402
from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (  # noqa: E402
    cutedsl_bf16_gemm,
    use_cutedsl_bf16_gemm,
)

SHAPES = [(n, k) for n in [1024, 2624, 6144] for k in [2048, 6144]] + [(2048, 4096)]
NUM_TOKENS = get_ci_test_range(list(range(1, 33)), [1, 15, 16, 32])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_cutedsl_bf16_gemm(num_tokens, k, n, has_bias):
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")

    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") if has_bias else None
    if bias is not None:
        bias.requires_grad_(True)

    with torch.no_grad():
        out = cutedsl_bf16_gemm(x, weight, bias)
    assert out.shape == (num_tokens, n)
    assert out.dtype == torch.bfloat16

    ref = x.float() @ weight.float().T
    if bias is not None:
        ref = ref + bias.detach().float()
    torch.testing.assert_close(out, ref.bfloat16(), rtol=2e-2, atol=2.5)


@pytest.mark.parametrize("n, k", [(7168, 1536), (1536, 7168), (1024, 2048)])
def test_empty_batch_not_tgv_eligible(n, k):
    """DP-attention idle groups run a 0-token dummy forward to keep the
    mlp-sync lockstep; every m == 0 shape must route to cuBLAS."""
    assert not use_cutedsl_bf16_gemm(0, n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("has_bias", [False, True])
def test_cutedsl_bf16_gemm_empty_batch(has_bias):
    """Empty input must yield the empty [0, N] output, mirroring F.linear —
    launching TGV with a 0-CTA grid fails with CUDA_ERROR_INVALID_VALUE."""
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")

    n, k = 6144, 2048
    x = torch.empty(0, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") if has_bias else None

    out = cutedsl_bf16_gemm(x, weight, bias)
    torch.cuda.synchronize()
    assert out.shape == (0, n)
    assert out.dtype == torch.bfloat16


# Decode-sized projections whose K spans more k-tiles than the smem ring holds.
L2_PREFETCH_SHAPES = [
    (4120, 5120),
    (1536, 5120),
    (8704, 5120),
    (5120, 4352),
    (6144, 6144),
]


@pytest.mark.parametrize("n,k", L2_PREFETCH_SHAPES)
@pytest.mark.parametrize("num_tokens", [1, 8, 16])
def test_tgv_l2_prefetch_is_exact(num_tokens, k, n):
    """The weight L2 prefetch issued during the PDL wait only moves bytes into
    L2: the output is bitwise the output without it, and the PDL launch does
    compile the prefetching kernel."""
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")
    gen = torch.Generator(device="cuda").manual_seed(1000 * num_tokens + n + k)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    tactic = tgv._pick_tactic(num_tokens, n, k)
    outs = []
    for l2_prefetch in (True, False):
        out = torch.empty(num_tokens, n, dtype=torch.bfloat16, device="cuda")
        tgv._run_tgv(
            x, weight.t(), None, out, pdl=True, tactic=tactic, l2_prefetch=l2_prefetch
        )
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])
    cta_m = tgv._resolve_tactic(tactic)[0]
    prefetching = [
        key
        for key in tgv._TGV_CUTE_EXT_COMPILE_CACHE
        if key[2] == cta_m and key[-2] > 0
    ]
    assert prefetching, "the PDL launch did not compile the L2-prefetching kernel"


def test_tgv_l2_prefetch_budget_is_derived_from_the_device():
    """Half of the L2 per SM, issued at about the streaming HBM rate."""
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")
    per_cta, per_clk = tgv._tgv_l2_prefetch_budget(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(
        f"L2 prefetch budget: {per_cta} B per CTA, {per_clk} B per SM clock ({props.name})"
    )
    assert per_cta * props.multi_processor_count <= props.L2_cache_size // 2
    # At least one k-tile of the smallest cta_m.
    assert per_cta >= 64 * tgv._TGV_CUTE_EXT_CTA_K * 2
    assert per_clk > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

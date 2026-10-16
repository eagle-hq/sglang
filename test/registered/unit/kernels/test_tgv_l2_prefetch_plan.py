"""The TGV GEMM's weight L2 prefetch window and pace come from the device.

The window is half of L2 over one CTA per SM, the pace the peak memory rate per
SM clock; a device that reports no clocks gets no prefetch. The prefetch also
needs PDL, a K-major weight and K beyond the smem ring. The decisions are pure
functions of the device properties, so the test needs no GPU.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

pytest.importorskip("cutlass")

from sglang.kernels.ops.gemm import cutedsl_bf16_gemm as tgv  # noqa: E402

CTA_K = tgv._TGV_CUTE_EXT_CTA_K
RING_K = 8 * CTA_K  # the default tactic's smem ring (8 stages)


def _props(**overrides):
    props = dict(
        L2_cache_size=128 << 20,
        multi_processor_count=128,
        memory_clock_rate=4_000_000,  # kHz
        memory_bus_width=8192,  # bits
        clock_rate=2_000_000,  # kHz
        is_integrated=False,
    )
    props.update(overrides)
    return SimpleNamespace(**props)


def _budget(props):
    tgv._l2_prefetch_budget.cache_clear()
    with patch.object(tgv.torch.cuda, "get_device_properties", return_value=props):
        try:
            return tgv._l2_prefetch_budget(0)
        finally:
            tgv._l2_prefetch_budget.cache_clear()


def _plan(budget=(512 << 10, 4096), **overrides):
    args = dict(pdl=True, k_major=True, k=2 * RING_K, cta_m=64, num_ab_stage=8)
    args.update(overrides)
    return tgv._l2_prefetch_plan(budget=budget, **args)


class TestTgvL2PrefetchPlan(CustomTestCase):
    def test_budget_follows_the_device(self):
        # 128 MiB L2 / 2 / 128 SMs; 2 x 4 GHz x 1 KiB = 8.2 TB/s over 2 GHz.
        self.assertEqual(_budget(_props()), (512 << 10, 4096))
        self.assertEqual(
            _budget(_props(L2_cache_size=64 << 20, memory_bus_width=4096)),
            (256 << 10, 2048),
        )
        self.assertEqual(_budget(_props(is_integrated=True)), (512 << 10, 2048))

    def test_no_prefetch_without_clocks(self):
        for field in ("memory_clock_rate", "memory_bus_width", "clock_rate"):
            self.assertEqual(_budget(_props(**{field: 0})), (0, 0), field)
        self.assertEqual(_plan(budget=(0, 0)), (0, 1))

    def test_window_and_pace(self):
        tile_bytes = 64 * CTA_K * 2
        tiles, clk = _plan()
        self.assertEqual(tiles, (512 << 10) // tile_bytes)
        self.assertEqual(clk, tile_bytes // 4096)
        # Rounded up: never faster than the memory rate.
        self.assertEqual(_plan(budget=(512 << 10, 3775))[1], 5)
        self.assertEqual(_plan(cta_m=128)[0], (512 << 10) // (2 * tile_bytes))

    def test_no_prefetch_when_a_precondition_fails(self):
        self.assertEqual(_plan(pdl=False), (0, 1))
        self.assertEqual(_plan(k_major=False), (0, 1))
        self.assertEqual(_plan(k=RING_K), (0, 1), "K fits the smem ring")


if __name__ == "__main__":
    unittest.main()

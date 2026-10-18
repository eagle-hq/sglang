"""Routing of bf16 GEMMs between TGV and cuBLAS (``use_cutedsl_bf16_gemm``) - CPU only."""

import unittest

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm_routing import use_cutedsl_bf16_gemm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCutedslBf16GemmRouting(CustomTestCase):
    def test_decode_batches_route_to_tgv(self):
        """Decode-sized batches run TGV, also in the region rules' cuBLAS areas
        (k < 2048, k > 6144, n > 8192), where TGV measured faster."""
        for m in (1, 7, 16):
            for n, k in (
                (5120, 1536),
                (8704, 5120),
                (34816, 5120),
                (4096, 14336),
                (7168, 768),
            ):
                with self.subTest(m=m, n=n, k=k):
                    self.assertTrue(use_cutedsl_bf16_gemm(m, n, k))

    def test_decode_rule_keeps_narrow_outputs_and_measured_ranges(self):
        self.assertFalse(use_cutedsl_bf16_gemm(1, 576, 7168))  # narrow n: cuBLAS wins
        self.assertFalse(
            use_cutedsl_bf16_gemm(8, 2304, 8192)
        )  # its measured range starts at 40
        self.assertFalse(use_cutedsl_bf16_gemm(17, 5120, 1536))


if __name__ == "__main__":
    unittest.main()

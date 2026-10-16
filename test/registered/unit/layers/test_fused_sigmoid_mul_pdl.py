"""fused_sigmoid_mul launches with PDL where the GPU supports it.

The attention output gate sits between the attention kernel and the
PDL-launched o_proj GEMM; launched without PDL it broke that chain. The
launch is checked with a recording stand-in for the Triton kernel, so the
test needs no GPU.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.elementwise import elementwise
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


class TestFusedSigmoidMulPDL(CustomTestCase):
    def _launch(self, pdl: bool) -> dict:
        kernel = _RecordingKernel()
        attn = torch.zeros(3, 8 * 16)
        gate = torch.zeros(3, 8, 32)[:, :, :16]  # strided 3-D gate
        with (
            patch.object(elementwise, "_fused_sigmoid_mul_kernel", kernel),
            patch.object(elementwise, "is_arch_support_pdl", return_value=pdl),
        ):
            elementwise.fused_sigmoid_mul(attn, gate)
        self.assertEqual(len(kernel.calls), 1)
        return kernel.calls[0][2]

    def test_pdl_launch_waits_in_kernel(self):
        kwargs = self._launch(pdl=True)
        self.assertIs(kwargs.get("launch_pdl"), True)
        self.assertIs(kwargs.get("USE_PDL"), True)

    def test_no_pdl_without_support(self):
        kwargs = self._launch(pdl=False)
        self.assertNotIn("launch_pdl", kwargs)
        self.assertIs(kwargs.get("USE_PDL"), False)


if __name__ == "__main__":
    unittest.main()

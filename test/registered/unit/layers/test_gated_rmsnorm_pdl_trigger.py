"""The gated RMSNorm kernel releases its PDL dependents before its main loads.

Under PDL the GEMM after the GDN gated norm (out_proj) may only start once this
kernel calls ``gdc_launch_dependents``. Calling it at kernel entry lets that
GEMM stream its weights while the norm runs; its own ``gdc_wait`` still waits
for this whole grid, so results do not change. Checked on the kernel source,
so the test needs no GPU.
"""

import ast
import importlib.util
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

KERNEL = "_layer_norm_fwd_1pass_kernel"


def _call_lines(fn: ast.FunctionDef, attr: str) -> list:
    return sorted(
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    )


class TestGatedRMSNormPDLTrigger(CustomTestCase):
    def setUp(self):
        root = Path(importlib.util.find_spec("sglang").origin).parent
        path = root / "kernels/ops/attention/fla/layernorm_gated.py"
        tree = ast.parse(path.read_text())
        fns = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == KERNEL
        ]
        self.assertEqual(len(fns), 1)
        self.kernel = fns[0]

    def test_trigger_follows_wait_and_precedes_loads(self):
        waits = _call_lines(self.kernel, "gdc_wait")
        triggers = _call_lines(self.kernel, "gdc_launch_dependents")
        loads = _call_lines(self.kernel, "load")
        self.assertEqual(len(waits), 1)
        self.assertEqual(len(triggers), 1, "trigger exactly once")
        self.assertLess(waits[0], triggers[0])
        self.assertLess(triggers[0], loads[0], "trigger before the first load")


if __name__ == "__main__":
    unittest.main()

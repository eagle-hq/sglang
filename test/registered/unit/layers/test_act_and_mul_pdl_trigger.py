"""act_and_mul releases its PDL dependents at kernel entry.

The down projection after SiLU-and-mul is PDL-launched; it can only start
loading weights once this kernel triggers. Triggering at entry instead of
after the store overlaps that load with the activation; the dependent's
PDLWaitPrimary still waits for the whole grid, so results are unchanged.
Checked on the CUDA source, so the test needs no GPU.
"""

import importlib.util
import re
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _kernel_body(source: str, name: str) -> str:
    start = source.index(f"__global__ void {name}(")
    depth, i = 0, source.index("{", start)
    for j in range(i, len(source)):
        depth += {"{": 1, "}": -1}.get(source[j], 0)
        if depth == 0:
            return source[i : j + 1]
    raise AssertionError(f"unbalanced body of {name}")


class TestActAndMulPDLTrigger(CustomTestCase):
    def test_trigger_before_wait_and_loads(self):
        root = Path(importlib.util.find_spec("sglang").origin).parent
        source = (root / "kernels/jit/csrc/elementwise/activation.cuh").read_text()
        body = _kernel_body(source, "act_and_mul_kernel")
        triggers = [m.start() for m in re.finditer(r"PDLTriggerSecondary<", body)]
        self.assertEqual(len(triggers), 1, "trigger exactly once")
        self.assertLess(triggers[0], body.index("PDLWaitPrimary<"))
        self.assertLess(triggers[0], body.index("load_as<"))


if __name__ == "__main__":
    unittest.main()

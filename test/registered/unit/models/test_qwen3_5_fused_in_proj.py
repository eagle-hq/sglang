"""Qwen3.5 GDN layers run one GEMM for both input projections on CUDA.

``finalize_fused_in_proj`` packs ``in_proj_qkvz`` and ``in_proj_ba`` into one
weight and ``_forward_input_proj`` runs that single GEMM for decode / verify
sized batches. Bug: the model's pre-capture hook called it on ROCm only, so on
CUDA b / a kept coming from a second GEMM on a side stream that every GDN layer
waited for, and the dense vision-language wrapper (Qwen3_5ForConditionalGeneration)
did not forward the hook at all. Guards: on CUDA the hook, called on that
wrapper, packs the layer, and the packed path returns the values of the separate
projections without running the separate b / a GEMM.
"""

import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import qwen3_5
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN, QKVZ, BA = 64, 96, 8


class _Linear(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(rows, HIDDEN, dtype=torch.bfloat16), requires_grad=False
        )
        self.bias = None
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x):
        return F.linear(x, self.weight), None


def _gdn():
    gdn = qwen3_5.Qwen3_5GatedDeltaNet.__new__(qwen3_5.Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(gdn)
    gdn.config = types.SimpleNamespace(model_type="qwen3_5_text")
    gdn.in_proj_qkvz, gdn.in_proj_ba = _Linear(QKVZ), _Linear(BA)
    gdn._fused_in_proj_weight = None
    gdn._fused_in_proj_qkvz_width = 0
    gdn.alt_stream = None
    gdn._fused_input_proj_cpu_enabled = types.SimpleNamespace(value=False)
    return gdn


class TestQwen35FusedInProjOnCuda(CustomTestCase):
    def setUp(self):
        for patcher in (
            patch.object(qwen3_5, "_is_cuda", True),
            patch.object(qwen3_5, "_use_aiter", False),
            patch.object(
                qwen3_5,
                "get_lora",
                return_value=types.SimpleNamespace(enable_lora=False, lora_paths=None),
            ),
            # The GEMM backend is the boundary; the CPU stand-in is F.linear.
            patch.object(qwen3_5, "bf16_gemm_dispatch", F.linear),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_prepare_hook_packs_and_the_packed_gemm_matches(self):
        torch.manual_seed(0)
        gdn = _gdn()
        want = [m.weight.data.clone() for m in (gdn.in_proj_qkvz, gdn.in_proj_ba)]
        # Through the wrapper Qwen3.5 checkpoints load as (dense, with vision).
        lm = types.SimpleNamespace(modules=lambda: [gdn])
        lm.prepare_before_cuda_graph_capture = types.MethodType(
            qwen3_5.Qwen3_5ForCausalLM.prepare_before_cuda_graph_capture, lm
        )
        model = types.SimpleNamespace(model=lm)
        qwen3_5.Qwen3_5ForConditionalGeneration.prepare_before_cuda_graph_capture(
            model, None
        )
        self.assertIsNotNone(gdn._fused_in_proj_weight)

        x = torch.randn(16, HIDDEN, dtype=torch.bfloat16)
        with patch.object(
            gdn.in_proj_ba, "forward", side_effect=AssertionError("separate b/a GEMM")
        ):
            qkvz, ba = gdn._forward_input_proj(x)
        torch.testing.assert_close(qkvz, F.linear(x, want[0]))
        torch.testing.assert_close(ba, F.linear(x, want[1]))


if __name__ == "__main__":
    unittest.main()

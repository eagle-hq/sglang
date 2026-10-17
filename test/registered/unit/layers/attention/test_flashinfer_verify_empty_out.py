"""A target verify on fa2 runs the paged prefill into an uninitialized output.

FlashInfer's paged ``run`` zero-fills the output it allocates: one fill kernel
per attention layer. A target verify writes every output row (each request's qo
range covers its draft tokens), so when FlashInfer's ``forward`` accepts ``out``
the backend passes an empty one shaped like ``q``; every other
forward keeps FlashInfer's own output, and an older FlashInfer gets no ``out``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import flashinfer_backend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROWS, HEADS, HEAD_DIM = 6, 4, 128


class _Wrapper:
    def forward(self, q, kv_cache, **kwargs):
        self.out = kwargs.get("out")
        return q.new_zeros(q.shape[:-1] + (HEAD_DIM,)) if self.out is None else self.out


def _forward_extend(mode, takes_out=True, prefill_backend="fa2"):
    wrapper = _Wrapper()
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            prefill_wrappers=[wrapper], use_ragged=False, multi_item_params=None
        ),
        _get_wrapper_idx=lambda layer: 0,
        prefill_uses_dequant_workspace=False,
        token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda layer_id: None),
        prefill_backend=prefill_backend,
    )
    layer = SimpleNamespace(
        logit_cap=0.0,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        tp_q_head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_id=0,
        scaling=0.1,
        sliding_window_size=-1,
        k_scale_float=None,
        v_scale_float=None,
    )
    q = torch.randn(ROWS, HEADS * HEAD_DIM, dtype=torch.bfloat16)
    with patch.object(
        flashinfer_backend, "_PAGED_PREFILL_FORWARD_TAKES_OUT", takes_out, create=True
    ):
        flashinfer_backend.FlashInferAttnBackend.forward_extend(
            backend, q, None, None, layer, SimpleNamespace(forward_mode=mode)
        )
    return wrapper.out


class TestFlashInferVerifyEmptyOut(CustomTestCase):
    def test_target_verify_on_fa2_passes_an_output_like_q(self):
        out = _forward_extend(ForwardMode.TARGET_VERIFY)
        self.assertEqual(tuple(out.shape), (ROWS, HEADS, HEAD_DIM))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_other_cases_keep_flashinfer_output(self):
        self.assertIsNone(_forward_extend(ForwardMode.EXTEND))
        self.assertIsNone(_forward_extend(ForwardMode.TARGET_VERIFY, takes_out=False))
        self.assertIsNone(
            _forward_extend(ForwardMode.TARGET_VERIFY, prefill_backend="trtllm-gen")
        )


if __name__ == "__main__":
    unittest.main()

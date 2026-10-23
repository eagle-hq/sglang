"""MRoPE positions of a decode forward - CPU only.

A speculative batch that decodes natively under the overlap scheduler has no
host lengths (``seq_lens_cpu`` is None): text-only rows take their positions
from the device lengths, as before; rows with multimodal inputs read the
lengths back.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _decode(seq_lens_cpu, mm_inputs):
    fb = ForwardBatch.__new__(ForwardBatch)
    fb.seq_lens = torch.tensor([5, 9])
    fb.seq_lens_cpu = seq_lens_cpu
    exec_ = SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target=None))
    with mock.patch.object(fbi, "get_exec", return_value=exec_):
        fb._compute_mrope_positions_decode(
            SimpleNamespace(device="cpu"), SimpleNamespace(multimodal_inputs=mm_inputs)
        )
    return fb.mrope_positions


class TestMropeDecodePositions(CustomTestCase):
    def test_text_rows_need_no_host_lengths(self):
        want = torch.tensor([[4, 8]] * 3)
        self.assertTrue(torch.equal(_decode(None, [None, None]), want))
        self.assertTrue(torch.equal(_decode(torch.tensor([5, 9]), [None, None]), want))

    def test_multimodal_rows_read_the_lengths_back(self):
        image = SimpleNamespace(
            mrope_positions=None, mrope_position_delta=torch.tensor([[-3]])
        )
        want = torch.tensor([[4, 5]] * 3)  # delta -3 shifts the image request
        self.assertTrue(torch.equal(_decode(None, [None, image]), want))


if __name__ == "__main__":
    unittest.main()

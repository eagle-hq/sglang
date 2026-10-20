"""FutureMap resolves host-published lengths without a device pull.

A publisher that commits a host-known token count per row (plain one-token
decode) hands the relay its host lengths; the next resolve must read them
without blocking the scheduler on the forward. Rows published device-only
still take the pulled path. Needs a GPU, no model:

    python -m pytest test/registered/unit/managers/test_future_map_host_lengths.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "FutureMap relay requires CUDA")
class TestFutureMapHostLengths(CustomTestCase):
    POOL = 8

    def setUp(self):
        device = torch.device("cuda")
        pool = SimpleNamespace(
            req_to_token=torch.zeros((self.POOL, 16), dtype=torch.int32, device=device)
        )
        self.future_map = FutureMap(device, SpeculativeAlgorithm.EAGLE, pool)
        self.indices_cpu = torch.tensor([3, 5], dtype=torch.int64)
        self.indices = self.indices_cpu.to(device)

    def _batch(self):
        return SimpleNamespace(
            spec_info=SimpleNamespace(future_indices=self.indices),
            req_pool_indices=self.indices,
            req_pool_indices_cpu=self.indices_cpu,
            seq_lens=None,
            seq_lens_cpu=None,
            seq_lens_sum=None,
        )

    def _resolve_without_device_pull(self, batch):
        with patch.object(
            self.future_map.fwd_prepare_d2h_stream,
            "synchronize",
            side_effect=AssertionError("resolve pulled seq_lens from the device"),
        ):
            self.future_map.resolve_seq_lens_cpu(batch)

    def test_host_published_lengths_resolve_without_a_device_pull(self):
        lengths_cpu = torch.tensor([11, 12], dtype=torch.int64)
        self.future_map.publish(
            self.indices,
            lengths_cpu.cuda(),
            future_indices_cpu=self.indices_cpu,
            new_seq_lens_cpu=lengths_cpu,
        )
        batch = self._batch()
        self._resolve_without_device_pull(batch)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [11, 12])
        self.assertEqual(batch.seq_lens_sum, 23)
        self.assertEqual(batch.seq_lens.tolist(), [11, 12])

    def test_device_only_publish_invalidates_host_rows(self):
        lengths_cpu = torch.tensor([11, 12], dtype=torch.int64)
        self.future_map.publish(
            self.indices,
            lengths_cpu.cuda(),
            future_indices_cpu=self.indices_cpu,
            new_seq_lens_cpu=lengths_cpu,
        )
        # A verify round commits an unknown accept count for the same rows.
        self.future_map.publish(
            self.indices,
            torch.tensor([14, 13], dtype=torch.int64, device="cuda"),
            future_indices_cpu=self.indices_cpu,
        )
        batch = self._batch()
        with self.assertRaisesRegex(AssertionError, "pulled seq_lens"):
            self._resolve_without_device_pull(batch)
        batch = self._batch()
        self.future_map.resolve_seq_lens_cpu(batch)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [14, 13])

    def test_publish_without_host_indices_forgets_every_host_row(self):
        lengths_cpu = torch.tensor([11, 12], dtype=torch.int64)
        self.future_map.publish(
            self.indices,
            lengths_cpu.cuda(),
            future_indices_cpu=self.indices_cpu,
            new_seq_lens_cpu=lengths_cpu,
        )
        other = torch.tensor([1], dtype=torch.int64, device="cuda")
        self.future_map.publish(other, torch.tensor([7], device="cuda"))
        batch = self._batch()
        with self.assertRaisesRegex(AssertionError, "pulled seq_lens"):
            self._resolve_without_device_pull(batch)


if __name__ == "__main__":
    unittest.main()

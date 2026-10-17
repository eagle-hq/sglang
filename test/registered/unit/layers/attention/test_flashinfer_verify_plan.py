"""FlashInfer target-verify planning - CPU only.

FlashInfer keys its target-verify CUDA-graph wrappers by (bs, num_tokens), so
a runner that captures several verify widths per batch size replays each
graph with the wrappers it was captured with.

Backends are built with ``__new__`` and only the attributes each method reads.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _flashinfer_backend():
    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.prefill_cuda_graph_metadata = {}
    backend.target_verify_cuda_graph_wrappers = {}
    backend.use_sliding_window_kv_pool = False
    backend.prefill_backend = "fa2"
    backend.dispatch_reason = None
    backend.indices_updater_prefill = mock.MagicMock()
    created = []

    def create(bs, use_custom_mask=False):
        wrappers = [SimpleNamespace(begin_forward=None)]
        created.append(wrappers)
        return wrappers

    backend._create_prefill_wrappers = create
    return backend, created


def _replay_view(bs, num_tokens, spec_info=None):
    return SimpleNamespace(
        batch_size=bs,
        req_pool_indices=torch.zeros(bs, dtype=torch.int64),
        seq_lens=torch.ones(bs, dtype=torch.int64),
        seq_lens_cpu=None,
        seq_lens_sum=bs,
        encoder_lens=None,
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=spec_info,
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        out_cache_loc=None,
        kv_loc_plan=None,
    )


class TestWidthKeyedVerifyWrappers(CustomTestCase):
    def test_each_verify_width_replays_its_own_wrappers(self):
        backend, created = _flashinfer_backend()
        # Capture widths 4 then 2 for bs=1 (widest first, as a provider does).
        backend._prepare_cuda_graph_metadata(1, 4, ForwardMode.TARGET_VERIFY, None)
        backend._prepare_cuda_graph_metadata(1, 2, ForwardMode.TARGET_VERIFY, None)
        wide, narrow = created

        backend.init_forward_metadata_out_graph(_replay_view(1, 4))
        used = backend.indices_updater_prefill.update.call_args.kwargs
        self.assertIs(used["prefill_wrappers"], wide)

        backend.init_forward_metadata_out_graph(_replay_view(1, 2))
        used = backend.indices_updater_prefill.update.call_args.kwargs
        self.assertIs(used["prefill_wrappers"], narrow)

    def test_one_width_per_bs_matches_the_bs_keyed_wrappers(self):
        backend, created = _flashinfer_backend()
        backend._prepare_cuda_graph_metadata(2, 8, ForwardMode.TARGET_VERIFY, None)
        backend.init_forward_metadata_out_graph(_replay_view(2, 8))
        used = backend.indices_updater_prefill.update.call_args.kwargs
        self.assertIs(used["prefill_wrappers"], backend.prefill_cuda_graph_metadata[2])
        self.assertIs(used["prefill_wrappers"], created[0])


if __name__ == "__main__":
    unittest.main()

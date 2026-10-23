"""The triton backend's CUDA-graph KV index buffers hold one context per request,
not one per token: a verify runner capturing 512 requests at 16 tokens each
must not reserve 16x the decode buffer."""

import unittest

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _backend(sliding_window_size=None):
    backend = object.__new__(TritonAttnBackend)
    backend.device = "meta"
    backend.dllm_attention = None
    backend.num_head = 4
    backend.max_kv_splits = 8
    backend.v_head_dim = 64
    backend.swa_v_head_dim = None
    backend.lean_total_programs = 2
    backend.lean_block_m = 16
    backend.max_context_len = 1024
    backend.is_draft_runner = False
    backend.skip_prefill = False
    backend.num_draft_tokens = 16
    backend.sliding_window_size = sliding_window_size
    backend.use_sliding_window_kv_pool = False
    return backend


class TestTritonGraphKvIndicesSize(unittest.TestCase):
    def test_kv_indices_hold_one_context_per_request(self):
        backend = _backend()
        backend.init_cuda_graph_state(max_bs=32, max_num_tokens=32 * 16)
        self.assertEqual(backend.cuda_graph_kv_indices.numel(), 32 * 1024)

    def test_window_kv_indices_hold_one_window_per_request(self):
        backend = _backend(sliding_window_size=128)
        backend.init_cuda_graph_state(max_bs=32, max_num_tokens=32 * 16)
        self.assertEqual(backend.cuda_graph_window_kv_indices.numel(), 32 * 128)

    def test_decode_sizes_are_unchanged(self):
        backend = _backend()
        backend.init_cuda_graph_state(max_bs=32, max_num_tokens=32)
        self.assertEqual(backend.cuda_graph_kv_indices.numel(), 32 * 1024)


if __name__ == "__main__":
    unittest.main()

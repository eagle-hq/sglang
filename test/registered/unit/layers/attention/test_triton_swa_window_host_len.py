"""The eager sliding-window gather sizes its index buffer from host lengths.

``update_sliding_window_buffer`` allocated ``window_kv_indices`` with the
device total ``window_kv_indptr[-1]``: a cudaStreamSynchronize. A speculative
draft plans its prefill extend right after the target's prefill is queued, so
that read parked the scheduler for the whole target prefill, and the host work
that overlap scheduling hides ran with the GPU idle. With the host-known sum of
``min(len, window)`` the call queues work and returns.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.triton_backend import update_sliding_window_buffer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

WINDOW = 4


class _Translator:
    """Writes each request's window as ``kv_start_idx + 0..len`` (no req_to_token)."""

    window_kv_kind = None

    def pack_read_stream(self, plan, *, seq_lens, indptr, out, kv_start_idx, **_):
        for i in range(len(seq_lens)):
            n = int(seq_lens[i])
            start = int(indptr[i])
            out[start : start + n] = kv_start_idx[i] + torch.arange(
                n, device=out.device
            )
        return True


def _gather(window_kv_lens_sum):
    lens = torch.tensor([0, 3, 9, 4], dtype=torch.int32, device="cuda")
    indptr = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    return update_sliding_window_buffer(
        indptr,
        _Translator(),
        None,
        torch.arange(len(lens), device="cuda"),
        WINDOW,
        lens,
        len(lens),
        device="cuda",
        token_to_kv_pool=SimpleNamespace(),
        window_kv_lens_sum=window_kv_lens_sum,
    )


def _sync_error(fn):
    """The RuntimeError torch raises if ``fn`` synchronizes, or None."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    except RuntimeError as exc:
        return exc
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    return None


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestTritonSwaWindowHostLen(CustomTestCase):
    def test_host_sum_gathers_the_device_sized_window(self):
        _, by_host, lens, start = _gather(0 + 3 + 4 + 4)
        _, by_device, _, _ = _gather(None)
        self.assertEqual(lens.tolist(), [0, 3, 4, 4])
        self.assertEqual(start.tolist(), [0, 0, 5, 0])
        self.assertEqual(by_host.tolist(), by_device.tolist())
        self.assertEqual(by_host.tolist(), [0, 1, 2, 5, 6, 7, 8, 0, 1, 2, 3])

    def test_allocation_does_not_wait_for_the_device(self):
        # Only the sizing is under test: the fake translator's own reads sync.
        lens = torch.tensor([3, 9], dtype=torch.int32, device="cuda")
        indptr = torch.zeros(3, dtype=torch.int32, device="cuda")

        class _Quiet(_Translator):
            def pack_read_stream(self, plan, **_):
                return True

        def gather(total):
            return lambda: update_sliding_window_buffer(
                indptr,
                _Quiet(),
                None,
                torch.arange(2, device="cuda"),
                WINDOW,
                lens,
                2,
                device="cuda",
                token_to_kv_pool=SimpleNamespace(),
                window_kv_lens_sum=total,
            )

        self.assertIsNone(_sync_error(gather(3 + 4)))
        self.assertIsNotNone(_sync_error(gather(None)))


if __name__ == "__main__":
    unittest.main()

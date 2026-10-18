"""A CUDA-graph replay builds its verify attention arrays in the wrapper's buffers - CPU only.

``generate_attn_arg_prefill`` of the speculative inputs takes optional output
buffers (``qo_indptr_buf``, ``kv_indptr_buf``, ``kv_indices_buf``): a FlashInfer
CUDA-graph wrapper planned by ``fast_prefill_plan`` passes its own input buffers,
and the plan then leaves those arrays in place instead of copying them.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_backend import fast_prefill_plan
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput, EagleVerifyInput
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_TOKENS = 4
SEQ_LENS = [7, 3]


class _Translator:
    """Packs request ``i``'s first ``seq_lens[i]`` req_to_token entries at ``indptr[i]``."""

    def __init__(self, req_to_token):
        self.req_to_token = req_to_token

    def pack_read_stream(
        self, plan, *, req_pool_indices, seq_lens, indptr, out, kind, kv_start_idx=None
    ):
        for i, n in enumerate(seq_lens.tolist()):
            start = int(indptr[i])
            out[start : start + n] = self.req_to_token[req_pool_indices[i], :n]
        return True


def _verify_input():
    return EagleVerifyInput(
        draft_token=torch.zeros(len(SEQ_LENS) * NUM_TOKENS, dtype=torch.int64),
        custom_mask=torch.ones(1024, dtype=torch.bool),
        positions=None,
        retrieve_index=None,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_cum_len=None,
        spec_steps=3,
        topk=1,
        draft_token_num=NUM_TOKENS,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        seq_lens_sum=sum(SEQ_LENS),
        seq_lens_cpu=torch.tensor(SEQ_LENS),
    )


def _draft_extend_input():
    spec = EagleDraftExtendInput.__new__(EagleDraftExtendInput)
    spec.num_correct_drafts = torch.zeros(len(SEQ_LENS), dtype=torch.int32)
    spec.num_tokens_per_req = NUM_TOKENS
    return spec


def _ngram_input():
    spec = NgramVerifyInput.__new__(NgramVerifyInput)
    spec.draft_token_num = NUM_TOKENS
    spec.device = torch.device("cpu")
    spec.custom_mask = torch.ones(1024, dtype=torch.bool)
    return spec


class TestVerifyArgsInGraphBuffers(CustomTestCase):
    def setUp(self):
        bs = len(SEQ_LENS)
        self.req_to_token = torch.arange(4 * 64, dtype=torch.int32).view(4, 64)
        self.req_pool_indices = torch.tensor([2, 0])
        self.seq_lens = torch.tensor(SEQ_LENS)
        # A graph wrapper's buffers hold 0 at kv_indptr[0] and stale values elsewhere.
        self.buffers = dict(
            qo_indptr_buf=torch.full((bs + 1,), -1, dtype=torch.int32),
            kv_indptr_buf=torch.tensor([0] + [-1] * bs, dtype=torch.int32),
            kv_indices_buf=torch.full((64,), -1, dtype=torch.int32),
        )

    def _args(self, spec, lens_sum, **buffers):
        return spec.generate_attn_arg_prefill(
            req_pool_indices=self.req_pool_indices,
            paged_kernel_lens=self.seq_lens,
            paged_kernel_lens_sum=lens_sum,
            translator=_Translator(self.req_to_token),
            plan=None,
            **buffers,
        )

    def _check(self, make_spec, lens_sum):
        kv_indices, kv_indptr, qo_indptr, _ = self._args(make_spec(), lens_sum)
        in_buf = self._args(make_spec(), lens_sum, **self.buffers)
        # The arrays are views of the buffers, with the values of fresh ones.
        self.assertEqual(
            in_buf[0].data_ptr(), self.buffers["kv_indices_buf"].data_ptr()
        )
        self.assertEqual(in_buf[1].data_ptr(), self.buffers["kv_indptr_buf"].data_ptr())
        self.assertEqual(in_buf[2].data_ptr(), self.buffers["qo_indptr_buf"].data_ptr())
        self.assertEqual(in_buf[1].tolist(), kv_indptr.tolist())
        self.assertEqual(in_buf[2].tolist(), qo_indptr.tolist())
        total = int(kv_indptr[-1])
        self.assertEqual(in_buf[0][:total].tolist(), kv_indices[:total].tolist())
        return kv_indptr, qo_indptr

    def test_eagle_verify(self):
        kv_indptr, qo_indptr = self._check(_verify_input, sum(SEQ_LENS))
        self.assertEqual(qo_indptr.tolist(), [0, 4, 8])
        self.assertEqual(
            kv_indptr.tolist(), [0, 11, 18]
        )  # widened by the 4 verify tokens

    def test_eagle_draft_extend(self):
        kv_indptr, qo_indptr = self._check(_draft_extend_input, sum(SEQ_LENS))
        self.assertEqual(qo_indptr.tolist(), [0, 4, 8])
        self.assertEqual(kv_indptr.tolist(), [0, 7, 10])  # lengths include the window

    def test_ngram_verify(self):
        kv_indptr, qo_indptr = self._check(_ngram_input, sum(SEQ_LENS))
        self.assertEqual(qo_indptr.tolist(), [0, 4, 8])
        self.assertEqual(kv_indptr.tolist(), [0, 11, 18])


class TestFastPrefillPlanLeavesBuiltArrays(CustomTestCase):
    def _wrapper(self):
        plans = []
        module = SimpleNamespace(plan=lambda *args: plans.append(args) or "plan-info")
        wrapper = SimpleNamespace(
            is_cuda_graph_enabled=True,
            _backend="fa2",
            _cached_module=module,
            _max_total_num_rows=None,
            device=torch.device("cpu"),
            _qo_indptr_buf=torch.zeros(3, dtype=torch.int32),
            _paged_kv_indptr_buf=torch.zeros(3, dtype=torch.int32),
            _paged_kv_indices_buf=torch.zeros(32, dtype=torch.int32),
            _paged_kv_last_page_len_buf=torch.ones(2, dtype=torch.int32),
            _float_workspace_buffer=None,
            _int_workspace_buffer=None,
            _pin_memory_int_workspace_buffer=None,
        )
        return wrapper, plans

    def _plan(self, wrapper, qo, kv_indptr, kv_indices, last_page_len):
        fast_prefill_plan(
            wrapper,
            qo,
            kv_indptr,
            kv_indices,
            last_page_len,
            8,
            8,
            128,
            1,
            q_data_type=torch.bfloat16,
            qo_indptr_host=qo.clone(),
            kv_indptr_host=kv_indptr.clone(),
            kv_lens_host=kv_indptr[1:] - kv_indptr[:-1],
            max_q_len=4,
            max_kv_len=11,
        )

    def test_built_in_place_and_copied(self):
        qo = torch.tensor([0, 4, 8], dtype=torch.int32)
        kv_indptr = torch.tensor([0, 11, 18], dtype=torch.int32)
        kv_indices = torch.arange(18, dtype=torch.int32)
        ones = torch.ones(2, dtype=torch.int32)

        copied, plans = self._wrapper()
        self._plan(copied, qo, kv_indptr, kv_indices, ones)

        built, _ = self._wrapper()
        built._qo_indptr_buf.copy_(qo)
        built._paged_kv_indptr_buf.copy_(kv_indptr)
        built._paged_kv_indices_buf[:18].copy_(kv_indices)
        self._plan(
            built,
            built._qo_indptr_buf,
            built._paged_kv_indptr_buf,
            built._paged_kv_indices_buf,
            built._paged_kv_last_page_len_buf,
        )
        for name in ("_qo_indptr_buf", "_paged_kv_indptr_buf", "_paged_kv_indices_buf"):
            self.assertEqual(
                getattr(built, name).tolist(), getattr(copied, name).tolist(), name
            )
        self.assertEqual(copied._paged_kv_indices_buf[:18].tolist(), list(range(18)))
        self.assertEqual(len(plans), 1)


if __name__ == "__main__":
    unittest.main()

"""EAGLE verify accepts the greedy ids the target forward returned - CPU only.

With ``SpecInput.greedy_tokens_only`` the logits processor returns the verify
rows' argmax (``LogitsProcessorOutput.next_token_ids``) and no logits.
``eagle_sample`` then feeds those ids to the greedy tree walk directly, and
refuses a batch that needs the logits (sampling, penalties, logit bias, a
grammar, a watermark, logprobs). The tree walk kernel is stubbed; only the
hand-off of the ids is under test.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

BS, WIDTH, VOCAB = 2, 4, 8


def _batch(**sampling_overrides):
    sampling = SimpleNamespace(
        is_all_greedy=True,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        has_watermark_candidates=False,
    )
    for name, value in sampling_overrides.items():
        setattr(sampling, name, value)
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        device="cpu",
        seq_lens=torch.tensor([5, 9]),
        return_logprob=False,
        sampling_info=sampling,
    )


def _verify_input():
    return SimpleNamespace(
        draft_token_num=WIDTH,
        max_tree_depth=WIDTH,
        tree_topk=1,
        draft_token=torch.arange(BS * WIDTH, dtype=torch.int64),
        retrieve_index=torch.arange(BS * WIDTH, dtype=torch.int64).view(BS, WIDTH),
        retrieve_next_token=torch.zeros(BS, WIDTH, dtype=torch.int64),
        retrieve_next_sibling=torch.zeros(BS, WIDTH, dtype=torch.int64),
    )


def _sample(logits_output, batch, **kwargs):
    """eagle_sample with the tree walk stubbed; returns the walk's keyword arguments."""
    calls = []

    def walk(**walk_kwargs):
        calls.append(walk_kwargs)
        return walk_kwargs["predicts"], walk_kwargs["accept_index"], walk_kwargs["accept_token_num"]

    with (
        mock.patch.object(eagle_utils, "verify_tree_greedy_func", side_effect=walk),
        mock.patch.object(
            eagle_utils,
            "get_spec",
            return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
        ),
    ):
        eagle_sample(_verify_input(), batch, logits_output, **kwargs)
    return calls


class TestEagleSampleGreedyIds(CustomTestCase):
    def test_greedy_ids_feed_the_tree_walk_without_logits(self):
        ids = torch.randint(0, VOCAB, (BS * WIDTH,))
        logits_output = SimpleNamespace(next_token_logits=None, next_token_ids=ids)
        calls = _sample(logits_output, _batch())
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(calls[0]["target_predict"], ids.view(BS, WIDTH))
        self.assertEqual(calls[0]["predicts"].shape, (BS * WIDTH,))
        self.assertEqual(calls[0]["accept_index"].shape, (BS, WIDTH))

    def test_logits_still_take_the_argmax(self):
        logits = torch.randn(BS * WIDTH, VOCAB)
        logits_output = SimpleNamespace(next_token_logits=logits, next_token_ids=None)
        calls = _sample(logits_output, _batch())
        torch.testing.assert_close(calls[0]["target_predict"], logits.argmax(-1).view(BS, WIDTH))

    def test_ids_are_refused_where_the_logits_are_needed(self):
        ids = torch.zeros(BS * WIDTH, dtype=torch.int64)
        logits_output = SimpleNamespace(next_token_logits=None, next_token_ids=ids)
        cases = {
            "non-greedy sampling": (_batch(is_all_greedy=False), {}),
            "penalties": (_batch(acc_additive_penalties=torch.zeros(BS, VOCAB)), {}),
            "logit bias": (_batch(logit_bias=torch.zeros(BS, VOCAB)), {}),
            "a grammar": (_batch(), {"grammar_mask": SimpleNamespace()}),
            "a watermark": (
                _batch(has_watermark_candidates=True),
                {"watermark_state": SimpleNamespace()},
            ),
        }
        logprob_batch = _batch()
        logprob_batch.return_logprob = True
        cases["logprobs"] = (logprob_batch, {})
        for need, (batch, kwargs) in cases.items():
            with self.subTest(need=need), self.assertRaisesRegex(ValueError, need):
                _sample(logits_output, batch, **kwargs)


if __name__ == "__main__":
    unittest.main()

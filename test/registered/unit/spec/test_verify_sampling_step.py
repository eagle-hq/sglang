"""The probability seam retains target filtering and host acceptance semantics."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import eagle_utils as utils


def test_probability_sampler_sees_filtered_targets_and_returns_bonus_lengths():
    from sglang.srt.layers import dp_attention
    from sglang.srt import runtime_context
    from sglang.srt.speculative import spec_utils

    batch = SimpleNamespace(
        device="cpu",
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([3]),
        sampling_info=SimpleNamespace(
            is_all_greedy=False,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            has_watermark_candidates=False,
            temperatures=torch.tensor([[2.0]]),
        ),
    )
    verify = SimpleNamespace(
        draft_token=torch.tensor([1, 2]),
        draft_token_num=2,
        max_tree_depth=2,
    )
    logits = torch.tensor([[0.0, 2.0, 4.0], [4.0, 2.0, 0.0]])
    coins, final = torch.tensor([[0.1, 0.2]]), torch.tensor([0.3])
    prediction = torch.tensor([2, 0], dtype=torch.int32)
    path = torch.tensor([[0, 1]], dtype=torch.int32)
    filtered = torch.tensor([[0.0, 0.25, 0.75], [0.75, 0.25, 0.0]])
    observed = []

    def filter_probs(probs, sampling, width, **kwargs):
        torch.testing.assert_close(probs, torch.softmax(logits / 2, dim=-1))
        return filtered

    def sample(probs, actual_coins, actual_final):
        observed.append(probs)
        torch.testing.assert_close(probs, filtered.view(1, 2, 3))
        assert actual_coins is coins and actual_final is final
        return prediction, torch.tensor([2], dtype=torch.int32), path

    with (
        patch.dict(
            "sys.modules",
            {
                "sgl_kernel": SimpleNamespace(
                    tree_speculative_sampling_target_only=None,
                    top_k_renorm_prob=None,
                    top_p_renorm_prob=None,
                )
            },
        ),
        patch.object(
            utils,
            "get_spec",
            return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
        ),
        patch.object(utils, "_verify_coins", return_value=(coins, final)),
        patch.object(utils, "_filter_verify_target_probs", side_effect=filter_probs),
        patch.object(dp_attention, "is_dp_attention_enabled", return_value=False),
        patch.object(
            runtime_context,
            "get_parallel",
            return_value=SimpleNamespace(tp_group=SimpleNamespace(world_size=1)),
        ),
        patch.object(spec_utils, "SIMULATE_ACC_LEN", 0),
    ):
        result = utils.eagle_sample(
            verify,
            batch,
            SimpleNamespace(next_token_logits=logits.clone()),
            sampling_step=sample,
        )
    assert len(observed) == 1
    assert result[0] is prediction and result[2] is path
    assert result[1].tolist() == [2]

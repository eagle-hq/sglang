"""State-commit extension routing without a GPU or optional ROCm packages."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention import hybrid_linear_attn_backend as module
from sglang.srt.layers.attention.linear.verify_extension import (
    LinearAttnVerifyExtension,
)


@pytest.mark.parametrize(
    "has_extension,has_parent,handled,tracking",
    [
        (False, True, False, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
    ],
)
def test_state_commit_only_replaces_ssm_scatter(
    has_extension, has_parent, handled, tracking
):
    state = torch.zeros(2, 4, 3, 8, 8)
    slots = torch.tensor([3, 1], dtype=torch.int32)
    rows = torch.tensor([1, 0], dtype=torch.int32)
    parents = torch.zeros(2, 8, dtype=torch.int32) if has_parent else None
    steps = torch.tensor([3, 5], dtype=torch.int32)
    extension = (
        SimpleNamespace(commit_state=Mock(return_value=handled))
        if has_extension
        else None
    )
    cache = SimpleNamespace(temporal=state)
    linear = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            mamba_cache_indices=slots, retrieve_parent_token=parents
        ),
        req_to_token_pool=SimpleNamespace(
            get_speculative_mamba2_params_all_layers=lambda: cache,
            mamba_pool=SimpleNamespace(replayssm_is_kda=False),
        ),
        verify_extension=extension,
        verify_intermediate_state_indices=rows,
        accept_lens_pool=None,
        _translate_mamba_indices=lambda value: value,
    )
    backend = module.HybridLinearAttnBackend.__new__(module.HybridLinearAttnBackend)
    backend.linear_attn_backend = linear
    backend._update_ple_state_after_mtp_verify = Mock()
    track = slots if tracking else None
    with (
        patch.object(module, "pp_spec_stable_rows_enabled", return_value=False),
        patch.object(module, "scatter_mamba_states_after_mtp_verify") as scatter,
    ):
        backend.update_mamba_state_after_mtp_verify(
            steps, track, steps if tracking else None, None
        )
    called = has_extension and has_parent and not tracking
    if has_extension:
        assert extension.commit_state.call_count == int(called)
        if called:
            args = extension.commit_state.call_args.args
            assert args[0] is state
            for actual, expected in zip(args[1:], (slots, rows, steps, parents)):
                torch.testing.assert_close(actual, expected)
    assert scatter.call_args.kwargs["skip_ssm"] is (called and handled)
    assert scatter.call_args.args[:1] == (cache,)
    backend._update_ple_state_after_mtp_verify.assert_called_once()


def test_default_extension_keeps_snapshot_commits():
    assert (
        LinearAttnVerifyExtension().commit_state(None, None, None, None, None) is False
    )

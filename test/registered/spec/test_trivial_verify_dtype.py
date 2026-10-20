"""The one-node verify input must satisfy the tree kernel's int64 token ABI."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_trivial_verify_accepts_bonus_token_dtypes(batch_size, dtype):
    device = torch.device("cuda")
    bonus = torch.arange(batch_size, dtype=dtype, device=device) + 10
    worker = SimpleNamespace(
        topk=1,
        device=device,
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(verify_mask=None, max_context_len=16)
            )
        ),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=SimpleNamespace(bonus_tokens=bonus),
        seq_lens=torch.full((batch_size,), 5, dtype=torch.int32, device=device),
        seq_lens_sum=5 * batch_size,
    )
    verify = EAGLEWorkerV2._build_trivial_verify_input(worker, batch)
    assert verify.draft_token.dtype == torch.int64
    torch.testing.assert_close(verify.draft_token, bonus.long())
    predicts = torch.empty(batch_size, dtype=torch.int32, device=device)
    accept_index = torch.empty((batch_size, 1), dtype=torch.int32, device=device)
    accepted = torch.empty(batch_size, dtype=torch.int32, device=device)
    target = torch.arange(batch_size, dtype=torch.int64, device=device)[:, None] + 100
    verify_tree_greedy_func(
        predicts,
        accept_index,
        accepted,
        verify.draft_token[:, None],
        verify.retrieve_index,
        verify.retrieve_next_token,
        verify.retrieve_next_sibling,
        target,
    )
    torch.testing.assert_close(predicts, target[:, 0].int())
    torch.testing.assert_close(accepted, torch.zeros_like(accepted))
    torch.testing.assert_close(
        accept_index[:, 0], torch.arange(batch_size, dtype=torch.int32, device=device)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

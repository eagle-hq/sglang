"""Overlap scheduling settles the pending result before a batch whose custom
logit processors read the request history.

Under overlap, a batch is sampled before the previous batch's result has been
appended to ``req.output_ids``, so a processor reading ``__req__`` (thinking
budget, no-repeat n-gram) saw one token too few. ``is_disable_overlap_for_batch``
must return True for such batches while a result is pending.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _batch(*, mode: str, custom: bool, spec: bool = False):
    return SimpleNamespace(
        is_extend_in_batch=mode == "extend",
        forward_mode=SimpleNamespace(
            is_extend=lambda: mode == "extend",
            is_decode=lambda: mode == "decode",
        ),
        spec_algorithm=SimpleNamespace(is_none=lambda: not spec),
        grammar_needs_sync=lambda: False,
        sampling_info=SimpleNamespace(has_custom_logit_processor=custom),
    )


def _scheduler(pending: int):
    scheduler = object.__new__(Scheduler)
    scheduler.require_mlp_sync = False
    scheduler.result_queue = [object()] * pending
    return scheduler


@patch(
    "sglang.srt.managers.scheduler.envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get",
    return_value=False,
)
class TestOverlapCustomLogitProcessorSync(CustomTestCase):
    def test_custom_processor_batch_waits_for_pending_result(self, _):
        last = _batch(mode="decode", custom=False)
        for mode in ("decode", "extend"):
            for spec in (False, True):
                batch = _batch(mode=mode, custom=True, spec=spec)
                self.assertTrue(_scheduler(1).is_disable_overlap_for_batch(batch, last))

    def test_other_batches_keep_overlap(self, _):
        last = _batch(mode="decode", custom=False)
        plain = _batch(mode="decode", custom=False)
        self.assertFalse(_scheduler(1).is_disable_overlap_for_batch(plain, last))
        custom = _batch(mode="decode", custom=True)
        self.assertFalse(_scheduler(0).is_disable_overlap_for_batch(custom, last))
        idle = _batch(mode="idle", custom=True)
        self.assertFalse(_scheduler(1).is_disable_overlap_for_batch(idle, last))
        self.assertFalse(_scheduler(1).is_disable_overlap_for_batch(None, last))


if __name__ == "__main__":
    unittest.main()

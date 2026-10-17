"""Unit tests for UNO allocation sizing."""

import unittest

from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import _REGISTRY, CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestUnoAllocationSizing(CustomTestCase):
    def test_page_size_one_row_covers_decode_reserve(self):
        with (
            get_context().override_server_args(
                speculative_algorithm="UNO",
                speculative_num_draft_tokens=8,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
        ):
            self.assertEqual(get_alloc_len_per_decode(), 9)
            self.assertEqual(get_alloc_reserve_per_decode(), 18)
            self.assertGreaterEqual(
                get_req_to_token_extra_context_len(),
                get_alloc_reserve_per_decode(),
            )


class TestPluginRowHeadroom(CustomTestCase):
    def setUp(self):
        self._snapshot = _REGISTRY.copy()
        self.addCleanup(lambda: (_REGISTRY.clear(), _REGISTRY.update(self._snapshot)))

    def test_plugin_headroom_only_raises_the_row(self):
        """The plugin's per-page-size headroom widens the row past the default,
        and a smaller request never shrinks it."""
        seen_page_sizes = []

        class _Algo(CustomSpecAlgo):
            headroom = 0

            def req_to_token_extra_len(self, page_size):
                seen_page_sizes.append(page_size)
                return _Algo.headroom

        SpeculativeAlgorithm.register("MY_ROWS", spec_class=_Algo)(lambda _: None)
        with get_context().override_server_args(
            speculative_algorithm="MY_ROWS",
            speculative_num_draft_tokens=4,
            page_size=1,
        ):
            default = get_req_to_token_extra_context_len()
            _Algo.headroom = default + 100
            self.assertEqual(get_req_to_token_extra_context_len(), default + 100)
        self.assertEqual(seen_page_sizes, [1, 1])


if __name__ == "__main__":
    unittest.main()

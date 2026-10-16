"""IndexKeyCache.move copies per-token index-K bytes in the page-major buffer.

Each buffer row is a page: ``page_size`` fp8 keys followed by their fp32
scales. ``move`` used to index rows by token slot, which with page_size > 1
copied whole wrong pages (or ran past the buffer). CPU only.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE, HEAD_DIM, QBLOCK, PAGES = 4, 8, 4, 6
SCALE_BYTES = HEAD_DIM // QBLOCK * 4


def _cache(page_size=PAGE, skip_layer=False):
    pool = SimpleNamespace(
        page_size=page_size,
        index_page_size=page_size,
        index_head_dim=HEAD_DIM,
        quant_block_size=QBLOCK,
    )
    cache = object.__new__(IndexKeyCache)
    cache.pool = pool
    row = page_size * (HEAD_DIM + SCALE_BYTES)
    gen = torch.Generator().manual_seed(0)
    cache.buffer = [
        torch.randint(0, 256, (PAGES, row), dtype=torch.uint8, generator=gen),
        torch.zeros(0, row, dtype=torch.uint8)
        if skip_layer
        else torch.randint(0, 256, (PAGES, row), dtype=torch.uint8, generator=gen),
    ]
    return cache


def _token(buf, loc, page_size=PAGE):
    page, slot = divmod(loc, page_size)
    key = buf[page, slot * HEAD_DIM : (slot + 1) * HEAD_DIM]
    s0 = page_size * HEAD_DIM + slot * SCALE_BYTES
    return torch.cat([key, buf[page, s0 : s0 + SCALE_BYTES]])


class TestIndexKeyCacheMove(CustomTestCase):
    def _check(self, page_size, src, tgt, skip_layer=False):
        cache = _cache(page_size, skip_layer)
        before = [b.clone() for b in cache.buffer]
        cache.move(torch.tensor(tgt), torch.tensor(src))
        for old, new in zip(before, cache.buffer):
            if old.numel() == 0:
                continue
            for s, t in zip(src, tgt):
                self.assertTrue(
                    torch.equal(_token(new, t, page_size), _token(old, s, page_size))
                )
            moved = set(tgt)
            for loc in range(PAGES * page_size):
                if loc not in moved:
                    self.assertTrue(
                        torch.equal(
                            _token(new, loc, page_size), _token(old, loc, page_size)
                        ),
                        f"slot {loc} changed",
                    )

    def test_paged_move(self):
        self._check(PAGE, src=[9, 10, 23], tgt=[4, 5, 6])

    def test_page_size_one(self):
        self._check(1, src=[3, 5], tgt=[0, 1])

    def test_skip_topk_layer_and_empty(self):
        self._check(PAGE, src=[13], tgt=[2], skip_layer=True)
        cache = _cache()
        before = cache.buffer[0].clone()
        cache.move(
            torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
        )
        self.assertTrue(torch.equal(cache.buffer[0], before))

    def test_pooled_layout_is_rejected(self):
        cache = _cache()
        cache.pool.index_page_size = PAGE // 2
        with self.assertRaises(NotImplementedError):
            cache.move(torch.tensor([1]), torch.tensor([2]))

    def test_page_size_one_moves_rows_in_any_layout(self):
        """With one token per row the row copy is right whatever the layout."""
        cache = _cache(1)
        cache.pool.index_page_size = 2
        before = cache.buffer[0].clone()
        cache.move(torch.tensor([0]), torch.tensor([3]))
        self.assertTrue(torch.equal(cache.buffer[0][0], before[3]))


if __name__ == "__main__":
    unittest.main()

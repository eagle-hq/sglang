# Modified by Eagle AI Corp.
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Split-KV target verify on ROCm against extend_attention_fwd (the kernel it replaces).

- Tree mode (topk > 1, the EAGLE-Harness): random draft trees, per-head and GQA-packed
  prefix stages, prefixes from tens of tokens to 32K, kv_indices padded like the HIP-graph
  buffer so the split counts resolve as in serving.
- Chain mode (topk 1: EAGLE chains, MTP): per-head and GQA-packed prefix stages.
- Both modes with the FlyDSL prefix stage (gfx950), which also drives the chain
  combine past 16 splits and the tree combine at its own split minimum.
Target shape: Qwen3.8-27B at TP1 (24 q / 4 kv heads x 256).

    python -m sglang.srt.speculative.test.test_rocm_verify_splitkv
"""

import random
import unittest

import torch

DEV = "cuda"
HQ, HKV, D = 24, 4, 256
GRAPH_INDICES = 64 * 36864


def _tree_rows(n, rng):
    parent = [-1] + [rng.randrange(0, i) for i in range(1, n)]
    m = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        j = i
        while j >= 0:
            m[i, j] = True
            j = parent[j]
    return m


def _case(bs, prefix_lens, l_ext, seed, tree):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    pool = sum(prefix_lens) + 64
    kb = torch.randn(pool, HKV, D, dtype=dtype, device=DEV)
    vb = torch.randn(pool, HKV, D, dtype=dtype, device=DEV)
    kvi = torch.zeros(
        max(GRAPH_INDICES, sum(prefix_lens)), dtype=torch.int64, device=DEV
    )
    kvi[: sum(prefix_lens)] = torch.randperm(pool, device=DEV)[: sum(prefix_lens)]
    kvp = torch.tensor(
        [0] + [sum(prefix_lens[: i + 1]) for i in range(bs)],
        dtype=torch.int32,
        device=DEV,
    )
    qo = torch.arange(0, (bs + 1) * l_ext, l_ext, dtype=torch.int32, device=DEV)
    q = torch.randn(bs * l_ext, HQ, D, dtype=dtype, device=DEV)
    k = torch.randn(bs * l_ext, HKV, D, dtype=dtype, device=DEV)
    v = torch.randn(bs * l_ext, HKV, D, dtype=dtype, device=DEV)
    mask = mask_indptr = None
    if tree:
        masks, starts = [], [0]
        for b in range(bs):
            rows = torch.ones(l_ext, prefix_lens[b] + l_ext, dtype=torch.bool)
            rows[:, prefix_lens[b] :] = _tree_rows(l_ext, rng)
            masks.append(rows.flatten())
            starts.append(starts[-1] + rows.numel())
        mask = torch.cat(masks).to(DEV)
        # int64, as TritonAttnBackend allocates it
        mask_indptr = torch.tensor(starts, dtype=torch.int64, device=DEV)
    return q, k, v, kb, vb, qo, kvp, kvi, mask, mask_indptr, l_ext


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None, "ROCm GPU only"
)
class TestRocmVerifySplitKV(unittest.TestCase):
    def setUp(self):
        from sglang.kernels.ops.attention import verify_splitkv as vsk
        from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

        self.vsk = vsk
        self.extend = extend_attention_fwd
        self.saved = (vsk.TREE_GQA, vsk.CHAIN_GQA, vsk.GQA_MIN_BS, vsk.FLYDSL)

    def tearDown(self):
        self.vsk.TREE_GQA, self.vsk.CHAIN_GQA, self.vsk.GQA_MIN_BS, self.vsk.FLYDSL = (
            self.saved
        )
        self.vsk._VK_CACHE.clear()

    def _check(self, c, tree):
        q, k, v, kb, vb, qo, kvp, kvi, mask, mask_indptr, l_ext = c
        ref = torch.empty_like(q)
        self.extend(
            q,
            k,
            v,
            ref,
            kb,
            vb,
            qo,
            kvp,
            kvi,
            mask,
            not tree,
            mask_indptr,
            l_ext,
            1.0,
            1.0,
            D**-0.5,
        )
        variants = [(False, False), (True, False)]
        if self.vsk._flydsl_supported():
            variants.append((True, True))
        for gqa, fly in variants:
            self.vsk.TREE_GQA = self.vsk.CHAIN_GQA = gqa
            self.vsk.GQA_MIN_BS = 1
            self.vsk.FLYDSL = fly
            self.vsk._VK_CACHE.clear()
            out = torch.empty_like(q)
            ran = self.vsk.verify_splitkv_fwd(
                q,
                k,
                v,
                out,
                kb,
                vb,
                qo,
                kvp,
                kvi,
                mask,
                not tree,
                mask_indptr,
                l_ext,
                1.0,
                1.0,
                D**-0.5,
                tree_mask=tree,
            )
            self.assertTrue(ran)
            self.assertLess(
                (ref.float() - out.float()).abs().max().item(),
                3e-2,
                f"gqa={gqa} fly={fly}",
            )

    def test_tree(self):
        cases = [
            (1, [37], 16),
            (1, [1000], 16),
            (1, [8192], 16),
            (1, [32768], 16),
            (1, [32768], 64),
            (4, [100, 2000, 8191, 517], 16),
            (4, [8192] * 4, 64),
            (16, [1024] * 16, 16),
        ]
        for i, (bs, lens, l_ext) in enumerate(cases):
            with self.subTest(bs=bs, prefix=lens[0], l_ext=l_ext):
                self._check(_case(bs, lens, l_ext, i, tree=True), tree=True)

    def test_chain(self):
        for i, (bs, prefix) in enumerate(
            [(1, 300), (1, 32768), (4, 1024), (16, 1024), (64, 1024), (16, 8192)]
        ):
            with self.subTest(bs=bs, prefix=prefix):
                self._check(
                    _case(bs, [prefix] * bs, 4, 100 + i, tree=False), tree=False
                )


if __name__ == "__main__":
    unittest.main()

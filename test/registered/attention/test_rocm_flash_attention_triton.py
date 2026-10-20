# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Triton flash_attn_with_kvcache / flash_attn_varlen_func that FlashAttentionBackend
(and the EAGLE-Harness fa3 draft path) runs on ROCm, against fp32 torch: paged (page
sizes 1 / 16 / 64, scattered blocks), contiguous and dense-varlen KV; GQA; causal,
left window, softcap; empty KV rows; LSE in FA3's [H, total_q] layout; the split-KV path
(the EAGLE-4 draft step's shapes: 40-head MHA, 4 tree rows, a 4096-key window) against the
one-program-per-head path, with a graph-padding query row.

    python -m sglang.srt.speculative.test.test_rocm_flash_attention_triton
"""

import itertools
import unittest

import torch

DEV = "cuda"


def _reference(q, keys, values, causal, window_left, scale, softcap):
    lq = q.shape[0]
    lk, hkv, _ = keys.shape
    g = q.shape[1] // hkv
    k = keys.float().repeat_interleave(g, dim=1)
    v = values.float().repeat_interleave(g, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
    if softcap:
        s = softcap * torch.tanh(s / softcap)
    qpos = (lk - lq + torch.arange(lq, device=q.device))[:, None]
    kpos = torch.arange(lk, device=q.device)[None, :]
    keep = torch.ones(lq, lk, dtype=torch.bool, device=q.device)
    if causal:
        keep &= kpos <= qpos
    if window_left >= 0:
        keep &= kpos >= qpos - window_left
    s = s.masked_fill(~keep[None], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None]).nan_to_num(0.0)
    return torch.einsum("hqk,khd->qhd", p, v), lse


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None, "ROCm GPU only"
)
class TestRocmFlashAttentionTriton(unittest.TestCase):
    def setUp(self):
        from sglang.kernels.ops.attention import flash_attention_triton as fa

        self.fa = fa
        torch.manual_seed(0)

    def _paged_case(
        self, q_lens, k_lens, hq, hkv, d, page, causal, window, softcap, pad_rows=0
    ):
        dtype = torch.bfloat16
        b = len(q_lens)
        cu_q = torch.tensor(
            [0] + list(itertools.accumulate(q_lens)), dtype=torch.int32, device=DEV
        )
        q = torch.randn(int(cu_q[-1]) + pad_rows, hq, d, dtype=dtype, device=DEV)
        ks = [torch.randn(n, hkv, d, dtype=dtype, device=DEV) for n in k_lens]
        vs = [torch.randn(n, hkv, d, dtype=dtype, device=DEV) for n in k_lens]
        max_blocks = max(1, max((n + page - 1) // page for n in k_lens))
        nblocks = b * max_blocks + 3
        k_cache = torch.randn(nblocks, page, hkv, d, dtype=dtype, device=DEV)
        v_cache = torch.randn(nblocks, page, hkv, d, dtype=dtype, device=DEV)
        perm = torch.randperm(nblocks, device=DEV)
        table = torch.zeros(b, max_blocks, dtype=torch.int32, device=DEV)
        used = 0
        for i, n in enumerate(k_lens):
            nb = (n + page - 1) // page
            blocks = perm[used : used + nb]
            used += nb
            table[i, :nb] = blocks.int()
            for j in range(n):
                k_cache[blocks[j // page], j % page] = ks[i][j]
                v_cache[blocks[j // page], j % page] = vs[i][j]
        out, lse = self.fa.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=torch.tensor(k_lens, dtype=torch.int32, device=DEV),
            page_table=table,
            cu_seqlens_q=cu_q,
            max_seqlen_q=max(q_lens),
            softmax_scale=d**-0.5,
            causal=causal,
            window_size=(window, 0 if causal else -1),
            softcap=softcap,
            return_softmax_lse=True,
        )
        for i in range(b):
            sl = slice(int(cu_q[i]), int(cu_q[i + 1]))
            ro, rl = _reference(q[sl], ks[i], vs[i], causal, window, d**-0.5, softcap)
            self.assertLess((out[sl].float() - ro).abs().max().item(), 2e-2)
            finite = torch.isfinite(rl)
            self.assertTrue(torch.equal(finite, torch.isfinite(lse[:, sl])))
            if finite.any():
                self.assertLess(
                    (lse[:, sl][finite] - rl[finite]).abs().max().item(), 1e-2
                )
        return q, k_cache, v_cache, table, cu_q, out, lse

    def test_paged(self):
        cases = [
            ([1, 5, 16], [37, 300, 4100], 40, 8, 128, 1, False, -1, 0.0),
            ([1, 5, 16], [37, 300, 4100], 40, 8, 128, 1, True, -1, 0.0),
            ([4, 4], [4096, 129], 40, 8, 128, 16, True, 63, 0.0),
            ([3, 1, 8], [3, 1, 8], 24, 4, 256, 1, False, -1, 0.0),
            ([2, 1], [0, 5], 40, 8, 128, 1, False, -1, 0.0),
            ([16, 7], [2000, 7], 24, 4, 256, 64, True, -1, 30.0),
        ]
        for case in cases:
            with self.subTest(case=case):
                self._paged_case(*case)

    def test_split_kv(self):
        # The draft step: 4 tree rows x 40 heads (MHA), a page-size-1 table over the window.
        cases = [
            ([4], [4096], 40, 40, 128, 1, False, 4095, 0.0, 0),
            ([4], [1000], 40, 40, 128, 1, False, -1, 0.0, 3),
            ([4, 4, 4, 4], [4096, 1000, 37, 0], 40, 40, 128, 1, False, 4095, 0.0, 0),
            ([1, 3], [5000, 2049], 40, 8, 128, 16, True, 1023, 0.0, 0),
        ]
        for case in cases:
            with self.subTest(case=case):
                q_lens, k_lens, hq = case[0], case[1], case[2]
                page = case[5]
                cap = max(1, max((n + page - 1) // page for n in k_lens)) * page
                self.assertGreater(
                    self.fa._num_splits(1, len(q_lens), hq, cap),
                    1,
                    "case does not exercise the split path",
                )
                q, kc, vc, table, cu_q, out, lse = self._paged_case(*case)
                # Bit-level agreement is not expected (summation order); the unsplit path
                # gives the same values to bf16 / fp32 rounding.
                self.fa.SPLIT_KV = False
                try:
                    ref_out, ref_lse = self.fa.flash_attn_with_kvcache(
                        q,
                        kc,
                        vc,
                        cache_seqlens=torch.tensor(
                            k_lens, dtype=torch.int32, device=DEV
                        ),
                        page_table=table,
                        cu_seqlens_q=cu_q,
                        max_seqlen_q=max(q_lens),
                        softmax_scale=case[4] ** -0.5,
                        causal=case[6],
                        window_size=(case[7], 0 if case[6] else -1),
                        return_softmax_lse=True,
                    )
                finally:
                    self.fa.SPLIT_KV = True
                n = int(cu_q[-1])
                self.assertLess(
                    (out[:n].float() - ref_out[:n].float()).abs().max().item(), 2e-2
                )
                fin = torch.isfinite(ref_lse[:, :n])
                self.assertTrue(torch.equal(fin, torch.isfinite(lse[:, :n])))
                if fin.any():
                    self.assertLess(
                        (lse[:, :n][fin] - ref_lse[:, :n][fin]).abs().max().item(), 1e-3
                    )

    def test_contiguous_cache(self):
        b, sq, s_cache, d = 2, 4, 700, 128
        lens = [700, 333]
        q = torch.randn(b, sq, 40, d, dtype=torch.bfloat16, device=DEV)
        k = torch.randn(b, s_cache, 8, d, dtype=torch.bfloat16, device=DEV)
        v = torch.randn(b, s_cache, 8, d, dtype=torch.bfloat16, device=DEV)
        out = self.fa.flash_attn_with_kvcache(
            q,
            k,
            v,
            cache_seqlens=torch.tensor(lens, dtype=torch.int32, device=DEV),
            causal=True,
        ).view(b, sq, 40, d)
        for i in range(b):
            ro, _ = _reference(
                q[i], k[i, : lens[i]], v[i, : lens[i]], True, -1, d**-0.5, 0.0
            )
            self.assertLess((out[i].float() - ro).abs().max().item(), 2e-2)

    def test_varlen_dense(self):
        q_lens, k_lens, d = [300, 17], [300, 17], 128
        cu_q = torch.tensor([0, 300, 317], dtype=torch.int32, device=DEV)
        q = torch.randn(317, 40, d, dtype=torch.bfloat16, device=DEV)
        ks = [torch.randn(n, 8, d, dtype=torch.bfloat16, device=DEV) for n in k_lens]
        vs = [torch.randn(n, 8, d, dtype=torch.bfloat16, device=DEV) for n in k_lens]
        out = self.fa.flash_attn_varlen_func(
            q,
            torch.cat(ks),
            torch.cat(vs),
            cu_q,
            cu_q.clone(),
            max_seqlen_q=300,
            max_seqlen_k=300,
            softmax_scale=d**-0.5,
            causal=True,
        )
        for i in range(2):
            sl = slice(int(cu_q[i]), int(cu_q[i + 1]))
            ro, _ = _reference(q[sl], ks[i], vs[i], True, -1, d**-0.5, 0.0)
            self.assertLess((out[sl].float() - ro).abs().max().item(), 2e-2)


if __name__ == "__main__":
    unittest.main()

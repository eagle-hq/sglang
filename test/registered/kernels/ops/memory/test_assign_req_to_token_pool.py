import unittest

import torch

from sglang.srt.mem_cache.allocation import (
    assign_req_to_token_pool,
    assign_req_to_token_pool_func,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestAssignReqToTokenPool(CustomTestCase):
    def test_batch_sizes_and_offsets_reuse_one_kernel(self):
        # Every decode step assigns a new batch size from wherever the
        # allocator's free list put the slots; neither may compile a kernel.
        kernel_cache = assign_req_to_token_pool.device_caches[
            torch.cuda.current_device()
        ][0]
        gen = torch.Generator(device="cuda").manual_seed(0)
        num_reqs, pool_len = 600, 96
        count = None
        for bs in (1, 2, 3, 17, 64, 100, 257, 512):
            for offset in range(4):
                req_pool_indices = torch.randperm(
                    num_reqs, device="cuda", generator=gen
                )[:bs].int()
                start = torch.randint(
                    0, 40, (bs,), device="cuda", generator=gen, dtype=torch.int32
                )
                end = start + torch.randint(
                    0, 50, (bs,), device="cuda", generator=gen, dtype=torch.int32
                )
                total = int((end - start).sum())
                free = torch.randperm(1 << 20, device="cuda", generator=gen)
                out_cache_loc = free[offset:][:total]
                req_to_token = torch.full(
                    (num_reqs, pool_len), -1, device="cuda", dtype=torch.int32
                )
                expected = req_to_token.clone()
                lo = 0
                for r, s, e in zip(
                    req_pool_indices.tolist(), start.tolist(), end.tolist()
                ):
                    expected[r, s:e] = out_cache_loc[lo : lo + e - s].int()
                    lo += e - s

                assign_req_to_token_pool_func(
                    req_pool_indices, req_to_token, start, end, out_cache_loc, bs
                )

                self.assertTrue(torch.equal(req_to_token, expected), (bs, offset))
                if count is None:
                    count = len(kernel_cache)
                self.assertEqual(len(kernel_cache), count, (bs, offset))


if __name__ == "__main__":
    unittest.main()

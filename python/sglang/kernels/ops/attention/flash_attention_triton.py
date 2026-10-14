# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
# Written by Aditya Nandakumar (@proaditya) at AMD. From eagle-hq/EAGLE-4 41b4d2dd, 15df42fb (PR #1).
"""Triton versions of flash_attn_with_kvcache / flash_attn_varlen_func for ROCm.

FlashAttentionBackend (and the EAGLE-Harness fa3 draft path built on it) call these two
FlashAttention-3 entry points. ROCm has no FA3 build, so flash_attention.py dispatches
here on HIP. The subset implemented is what the backend uses:

- q varlen ([total_q, Hq, D] with cu_seqlens_q) or batched ([B, Sq, Hq, D]);
- KV paged ([num_blocks, block_size, Hkv, D] + page_table [B, max_blocks]) or contiguous
  ([B, seqlen_cache, Hkv, D], no page_table) with cache_seqlens, or dense varlen
  ([total_k, Hkv, D] + cu_seqlens_k);
- GQA (Hq a multiple of Hkv); bottom-right aligned causal mask; left sliding window
  (window_size[0]); tanh softcap; softmax_scale;
- return_softmax_lse: natural-log LSE, [Hq, total_q] (FA3's varlen layout), -inf for
  rows that see no key (their output is 0).

Not implemented (asserted): appending new k/v into the cache, rotary, qv, descales,
cache_batch_idx / cache_leftpad, sinks, score_mod, a right window wider than causal.
Scheduling arguments (num_splits, pack_gqa, scheduler_metadata, sm_margin, ver) are
accepted and ignored.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fa_kvcache_fwd_kernel(
    Q, K, V, O, LSE,
    cu_seqlens_q, seqlens_k, cu_seqlens_k, page_table,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_ot, stride_oh,
    stride_lse_h,
    stride_pt_b,
    sm_scale, softcap,
    window_left,
    GQA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGED: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    kvh = h // GQA

    q_start = tl.load(cu_seqlens_q + b)
    q_len = tl.load(cu_seqlens_q + b + 1) - q_start
    if pid_m * BLOCK_M >= q_len:
        return
    k_len = tl.load(seqlens_k + b)
    if PAGED:
        k_base = 0
    else:
        k_base = tl.load(cu_seqlens_k + b)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < q_len
    d_mask = offs_d < HEAD_DIM

    q = tl.load(
        Q + (q_start + offs_m)[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    # Bottom-right aligned: query row i sits at key position k_len - q_len + i.
    q_pos = k_len - q_len + offs_m
    if CAUSAL:
        k_end = tl.minimum(k_len, k_len - q_len + (pid_m + 1) * BLOCK_M)
    else:
        k_end = k_len
    k_start = 0
    if USE_WINDOW:
        k_start = tl.maximum(k_len - q_len + pid_m * BLOCK_M - window_left, 0)
        k_start = (k_start // BLOCK_N) * BLOCK_N

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634  # log2(e): exp2 below

    for start_n in range(k_start, k_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < k_len
        if PAGED:
            block = tl.load(page_table + b * stride_pt_b + offs_n // PAGE_SIZE, mask=n_mask, other=0)
            k_off = block.to(tl.int64) * stride_kb + (offs_n % PAGE_SIZE) * stride_ks
            v_off = block.to(tl.int64) * stride_vb + (offs_n % PAGE_SIZE) * stride_vs
        else:
            k_off = (k_base + offs_n).to(tl.int64) * stride_ks
            v_off = (k_base + offs_n).to(tl.int64) * stride_vs
        k = tl.load(
            K + k_off[None, :] + kvh * stride_kh + offs_d[:, None],
            mask=n_mask[None, :] & d_mask[:, None],
            other=0.0,
        )
        qk = tl.dot(q, k)
        if USE_SOFTCAP:
            qk = qk * sm_scale
            qk = softcap * (2.0 / (1.0 + tl.exp(-2.0 * qk / softcap)) - 1.0)  # softcap * tanh(qk / softcap)
            qk = qk * 1.4426950408889634
        else:
            qk = qk * qk_scale
        keep = n_mask[None, :]
        if CAUSAL:
            keep = keep & (offs_n[None, :] <= q_pos[:, None])
        if USE_WINDOW:
            keep = keep & (offs_n[None, :] >= q_pos[:, None] - window_left)
        qk = tl.where(keep, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # Rows that have seen no key yet keep m = -inf; avoid (-inf) - (-inf).
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            V + v_off[:, None] + kvh * stride_vh + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    tl.store(
        O + (q_start + offs_m)[:, None] * stride_ot + h * stride_oh + offs_d[None, :],
        out.to(O.dtype.element_ty),
        mask=m_mask[:, None] & d_mask[None, :],
    )
    if WRITE_LSE:
        # m_i is in log2 units; LSE in natural log.
        lse = tl.where(l_i == 0.0, float("-inf"), (m_i + tl.log2(l_safe)) * 0.6931471805599453)
        tl.store(LSE + h * stride_lse_h + q_start + offs_m, lse, mask=m_mask)


def _launch(
    q, k, v, out, cu_seqlens_q, seqlens_k, cu_seqlens_k, page_table, max_seqlen_q,
    softmax_scale, causal, window_size, softcap, return_softmax_lse,
):
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[-2]
    assert num_heads % num_kv_heads == 0, (num_heads, num_kv_heads)
    left, right = window_size
    assert right in (-1, 0) or (causal and right <= 0), (
        f"window_size right={right} is not implemented (only causal / unbounded)"
    )
    batch = cu_seqlens_q.numel() - 1
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    if out is None:
        out = torch.empty_like(q)
    lse = (
        torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else out  # placeholder pointer, never written
    )
    paged = page_table is not None
    if paged:
        # [num_blocks, block_size, Hkv, D]
        stride_kb, stride_ks, stride_kh = k.stride(0), k.stride(1), k.stride(2)
        stride_vb, stride_vs, stride_vh = v.stride(0), v.stride(1), v.stride(2)
        page_size = k.shape[1]
        stride_pt_b = page_table.stride(0)
    else:
        # [total_k, Hkv, D]
        stride_kb, stride_ks, stride_kh = 0, k.stride(0), k.stride(1)
        stride_vb, stride_vs, stride_vh = 0, v.stride(0), v.stride(1)
        page_size = 1
        stride_pt_b = 0
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    block_m = 16 if max_seqlen_q <= 16 else 64
    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(max_seqlen_q, block_m), batch, num_heads)
    _fa_kvcache_fwd_kernel[grid](
        q, k, v, out, lse,
        cu_seqlens_q, seqlens_k, cu_seqlens_k if cu_seqlens_k is not None else seqlens_k,
        page_table if paged else seqlens_k,
        q.stride(0), q.stride(1),
        stride_kb, stride_ks, stride_kh,
        stride_vb, stride_vs, stride_vh,
        out.stride(0), out.stride(1),
        lse.stride(0) if return_softmax_lse else 0,
        stride_pt_b,
        softmax_scale, softcap if softcap else 1.0,
        left if left >= 0 else 0,
        GQA=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        PAGE_SIZE=page_size,
        PAGED=paged,
        CAUSAL=bool(causal),
        USE_WINDOW=left >= 0,
        USE_SOFTCAP=bool(softcap),
        WRITE_LSE=bool(return_softmax_lse),
        num_warps=4,
        num_stages=1,
    )
    return (out, lse) if return_softmax_lse else out


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    rotary_seqlens=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=None,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    sfq=None,
    sfk=None,
    sfv=None,
    rel_bias=None,
    rel_bias_prep_cache=None,
    ver=3,
    out=None,
    max_seqlen_k: Optional[int] = None,
):
    unsupported = {
        "k": k, "v": v, "qv": qv, "rotary_cos": rotary_cos, "cache_batch_idx": cache_batch_idx,
        "cache_leftpad": cache_leftpad, "q_descale": q_descale, "k_descale": k_descale,
        "v_descale": v_descale, "sinks": sinks, "score_mod": score_mod, "rel_bias": rel_bias,
        "attention_chunk": attention_chunk or None,
    }
    used = [name for name, value in unsupported.items() if value is not None]
    assert not used, f"flash_attn_with_kvcache (Triton, ROCm) does not implement {used}"
    cu_seqlens_k = None
    if page_table is None:
        # Contiguous cache [batch, seqlen_cache, Hkv, D]: the dense path over the flattened
        # cache, sequence b starting at b * seqlen_cache. (Not a page table with
        # seqlen_cache-token blocks: the block size is a kernel constexpr, so every cache
        # length would compile a new kernel.)
        batch_k, seqlen_cache = k_cache.shape[0], k_cache.shape[1]
        k_cache = k_cache.reshape(batch_k * seqlen_cache, k_cache.shape[2], k_cache.shape[3])
        v_cache = v_cache.reshape(batch_k * seqlen_cache, v_cache.shape[2], v_cache.shape[3])
        cu_seqlens_k = torch.arange(
            0, (batch_k + 1) * seqlen_cache, seqlen_cache, dtype=torch.int32, device=q.device
        )
    if cu_seqlens_q is None:
        batch, seqlen_q = q.shape[0], q.shape[1]
        q = q.reshape(batch * seqlen_q, q.shape[2], q.shape[3])
        cu_seqlens_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
        max_seqlen_q = seqlen_q
        if out is not None:
            out = out.view(batch * seqlen_q, out.shape[2], out.shape[3])
    if isinstance(cache_seqlens, int):
        cache_seqlens = torch.full((cu_seqlens_q.numel() - 1,), cache_seqlens, dtype=torch.int32, device=q.device)
    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
    return _launch(
        q, k_cache, v_cache, out, cu_seqlens_q, cache_seqlens, cu_seqlens_k, page_table, max_seqlen_q,
        softmax_scale, causal, window_size, softcap, return_softmax_lse,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=None,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
    out=None,
    **kwargs,
):
    unsupported = {
        "seqused_q": seqused_q, "qv": qv, "q_descale": q_descale, "k_descale": k_descale,
        "v_descale": v_descale, "sinks": sinks, "score_mod": score_mod, "attention_chunk": attention_chunk or None,
    }
    used = [name for name, value in unsupported.items() if value is not None]
    assert not used, f"flash_attn_varlen_func (Triton, ROCm) does not implement {used}"
    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
    if page_table is not None:
        seqlens_k = seqused_k if seqused_k is not None else (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)
        return _launch(
            q, k, v, out, cu_seqlens_q, seqlens_k, None, page_table, max_seqlen_q,
            softmax_scale, causal, window_size, softcap, return_softmax_lse,
        )
    seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)
    if seqused_k is not None:
        seqlens_k = seqused_k.to(torch.int32)
    return _launch(
        q, k, v, out, cu_seqlens_q, seqlens_k, cu_seqlens_k, None, max_seqlen_q,
        softmax_scale, causal, window_size, softcap, return_softmax_lse,
    )

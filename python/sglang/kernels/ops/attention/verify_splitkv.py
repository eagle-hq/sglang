"""Split-KV (flash-decode) attention for EAGLE speculative *verify*.

Two modes, chosen by the caller: a causal chain (speculative ``topk == 1``), and
with ``tree_mask=True`` a draft tree (``topk > 1``), whose draft-draft block
follows the custom tree mask (``_verify_combine_stage2_tree``). The prefix stage
is the same in both: every draft token sees the whole committed prefix.

On the Triton backend, EAGLE target-verify runs through the prefill
``extend_attention_fwd``, which loops the (long) prefix KV serially per
(sequence, head). With only a few draft-token queries, that leaves the GPU
memory system far under-utilized at long context. This kernel instead splits
the prefix KV across parallel programs (flash-decode style) and combines the
partials with a log-sum-exp merge, then handles the small causal draft-draft
block -- recovering memory bandwidth on the verify path.

Two Triton kernels:
  * ``_verify_prefix_stage1``: split-KV over the shared prefix. Applies the fp8
    dequant multipliers ``k_scale`` (on the QK score) and ``v_scale`` (on the
    prefix output), matching ``extend_attention_fwd``'s ``_fwd_kernel``
    (qk *= sm_scale * k_scale; acc += dot(p, v) * v_scale on the prefix loop;
    NO scaling on the draft-draft loop, whose K/V are the fresh bf16 draft
    tensors, not the fp8 pool). fp8 K/V buffers are handled by casting q to the
    buffer dtype before the dot (mirrors ``q.to(k.dtype)`` in the baseline).
  * ``_verify_combine_stage2``: combines the prefix splits (LSE merge) with the
    small causal draft-draft block and writes the output.

``verify_splitkv_fwd(...)`` takes the SAME positional args as
``extend_attention_fwd``; it runs the split-KV path when it can serve the case
bit-equivalently and returns True, otherwise returns False (doing nothing) so
the caller falls back to ``extend_attention_fwd``. Supported case: causal
(topk=1) verify with a constant per-sequence extend length, no sinks /
sliding-window / logit-cap / xai-temperature. Correctness is never violated.
"""

import functools
import os

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported, is_hip

_MIN_BLOCK_KV = 32

# AMD/CDNA-only Triton launch hints (waves_per_eu, matrix_instr_nonkdim); NVIDIA's
# Triton rejects these kwargs, so only pass them on ROCm. In production this kernel
# is dispatched only on AMD (see TritonAttnBackend); keeping it NV-safe lets the
# numerics test run on the CUDA CI lane.
_IS_HIP = is_hip()
_AMD_LAUNCH_KWARGS = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16} if _IS_HIP else {}

# Block-size config keyed on head_dim. The (BLOCK_N, num_warps) tile that best
# hides latency depends on head_dim: at head_dim=256 (Qwen3 family) a narrower
# BLOCK_N with more warps wins, since the 256-wide QK/PV tiles are register
# heavy. head_dim=256 is the value validated on MI350X; other head dims use a
# conservative default. Block size affects PERFORMANCE only, never correctness
# (any valid block size produces the same result).
DEFAULT_N_SPLITS = 8
DEFAULT_BLOCK_N = 32
DEFAULT_NUM_WARPS = 4
_BLOCK_CONFIG = {
    # head_dim: (BLOCK_N, num_warps)
    256: (32, 4),
}


def block_config(head_dim):
    """Return (BLOCK_N, num_warps) for a head_dim; default for untuned dims."""
    return _BLOCK_CONFIG.get(head_dim, (DEFAULT_BLOCK_N, DEFAULT_NUM_WARPS))


# ---------------------------------------------------------------------------
# Adaptive N_SPLITS.
# ---------------------------------------------------------------------------
# The prefix split-KV stage launches a (bs, h_q, N_SPLITS) grid; each (b,h,s)
# program handles kv_len_per_split = cdiv(cdiv(seqlen, N_SPLITS), MIN)*MIN keys.
# A fixed N_SPLITS=16 over-splits short/mid contexts (each split does too little
# work -> launch + reduction overhead dominates) and under-splits very long ones
# (too few parallel waves to saturate the device, raising tail latency on the
# slow split). Mirror the decode kernel's intent (decode_attention.py
# get_num_kv_splits): pick the split count per-dispatch from the representative
# sequence length, growing gradually with seqlen and capped at MAX.
#
# CRITICAL: this must be computed from STATIC shapes only (no .item()/.cpu()
# sync), because the verify/draft-extend step runs inside a captured HIP graph
# where a device->host copy raises hipErrorStreamCaptureUnsupported. We use the
# average prefix length = kv_indices.shape[0] / bs, which is a pure python int
# from tensor shapes -- no device read. N_SPLITS is then a power of two so the
# stage2 reduction tile (tl.arange(0, N_SPLITS)) stays cheap.
#
# Split-count bounds (internal constants). MAX=16 is the MI350X cap: 32
# oversubscribes the device and regresses, per tuning.
ADAPTIVE_SPLITS = True
MAX_N_SPLITS = 16
MIN_N_SPLITS = 4


# Tree mode packs the query heads of a KV head into one prefix program
# (_verify_prefix_stage1_gqa); SGLANG_SPLITKV_VERIFY_TREE_GQA=0 keeps the per-head
# prefix kernel (A/B and debugging).
TREE_GQA = os.environ.get("SGLANG_SPLITKV_VERIFY_TREE_GQA", "1") != "0"
# The same packed prefix stage for causal chains (topk 1) from GQA_MIN_BS; the chain
# keeps its combine stage and split sizing. SGLANG_SPLITKV_VERIFY_CHAIN_GQA=0: per-head.
CHAIN_GQA = os.environ.get("SGLANG_SPLITKV_VERIFY_CHAIN_GQA", "1") != "0"
GQA_MAX_ROWS = 128
GQA_TARGET_PROGRAMS = 512
GQA_MAX_N_SPLITS = 64
GQA_MIN_KEYS_PER_SPLIT = 256
GQA_BLOCK_N = 32
GQA_NUM_WARPS = 8
# Output columns per tree-combine program (power of two; the head dim covers 1+ tiles).
TREE_COMBINE_DV_TILE = 64
# Tree mode: at least this many prefix keys per split (sized in-kernel from the real
# prefix length, so a graph captured with the maximum split count stays cheap at short
# context), and the GQA-packed prefix stage from this batch size up (below it the
# per-head stage has more programs to fill the device).
TREE_MIN_SPLIT_KEYS = 256
GQA_MIN_BS = 4
# FlyDSL prefix stage (verify_prefix_flydsl.py) in place of both Triton prefix stages on
# gfx950 for bf16, head dim 256. Whole verify attention per layer on MI355X, split
# counts as under HIP graphs: 1.2-1.8x faster for batch >= 4 or prefix >= 8K (3.1x for a
# batch-1 32K chain); at batch 1 with a ~1K prefix 2-6 us slower. With it the GQA layout
# serves every batch size, chains take up to GQA_MAX_N_SPLITS splits (the chain combine
# merges them in chunks) and splits hold at least FLY_TREE_MIN_SPLIT_KEYS /
# FLY_CHAIN_MIN_SPLIT_KEYS keys. SGLANG_SPLITKV_VERIFY_FLYDSL=0: Triton.
FLYDSL = os.environ.get("SGLANG_SPLITKV_VERIFY_FLYDSL", "1") != "0"
FLY_TREE_MIN_SPLIT_KEYS = 256
FLY_CHAIN_MIN_SPLIT_KEYS = 64


@functools.lru_cache(maxsize=None)
def _flydsl_supported():
    if not (_IS_HIP and is_gfx95_supported()):
        return False
    try:
        from sglang.kernels.ops.attention import verify_prefix_flydsl  # noqa: F401
    except Exception:  # FlyDSL missing or too old
        return False
    return True


def use_flydsl(q, k_buffer, v_buffer, kv_indptr, kv_indices, k_scale, v_scale):
    """Static checks (no device reads, graph safe) for the FlyDSL prefix stage."""
    return (
        FLYDSL
        and _flydsl_supported()
        and q.dtype == k_buffer.dtype == v_buffer.dtype == torch.bfloat16
        and q.shape[2] == k_buffer.shape[2] == v_buffer.shape[2] == 256
        and q.stride(2) == 1 and q.stride(1) == 256
        and k_buffer.stride(2) == v_buffer.stride(2) == 1
        and k_buffer.stride(1) == v_buffer.stride(1) == 256
        and k_buffer.stride(0) == v_buffer.stride(0)
        and kv_indptr.dtype == torch.int32
        and kv_indices.dtype == torch.int64
        and float(k_scale) == 1.0
        and float(v_scale) == 1.0
    )


def gqa_rows(group, l_pad):
    """Rows per packed prefix program and the number of row chunks per KV head."""
    rows = min(GQA_MAX_ROWS, triton.next_power_of_2(group * l_pad))
    return rows, triton.cdiv(group * l_pad, rows)


def choose_n_splits_gqa(avg_seqlen, bs, h_kv, n_chunks):
    """Split count for the packed prefix grid (bs, h_kv * n_chunks, N_SPLITS): enough
    programs to fill the device (~GQA_TARGET_PROGRAMS), at least
    GQA_MIN_KEYS_PER_SPLIT keys per split, power of two in [MIN_N_SPLITS,
    GQA_MAX_N_SPLITS]. Static shapes only (HIP-graph safe), like choose_n_splits."""
    base = max(1, bs * h_kv * n_chunks)
    n = triton.next_power_of_2(triton.cdiv(GQA_TARGET_PROGRAMS, base))
    n = min(n, max(1, int(avg_seqlen) // GQA_MIN_KEYS_PER_SPLIT))
    n = 1 << (max(1, n).bit_length() - 1)  # round down to a power of two
    return max(MIN_N_SPLITS, min(GQA_MAX_N_SPLITS, n))


def choose_n_splits(avg_seqlen):
    """Pick N_SPLITS (power of two, in [MIN_N_SPLITS, MAX_N_SPLITS]) from the
    average prefix length. Tuned by the real-shape sweep (head_dim=256, BS*H_Q
    =128 base programs on ~132 CUs):

        ctx  <  4k -> 4   (short: extra splits add launch/reduction overhead)
        4k <= ctx < 8k -> 8   (sweet spot: best across 1k-16k in the sweep)
        ctx >= 8k      -> 16  (long: a few more splits help latency-bound tail)

    Never 32 (4096 grid blocks oversubscribes the device and regresses, per the
    sweep). Computed from a static shape (avg prefix = kv_indices.shape[0]/bs),
    so it is HIP-graph-capture safe (no device->host sync)."""
    if not ADAPTIVE_SPLITS:
        return DEFAULT_N_SPLITS
    s = int(avg_seqlen)
    if s < 4096:
        n = 4
    elif s < 8192:
        n = 8
    else:
        n = 16
    if n < MIN_N_SPLITS:
        n = MIN_N_SPLITS
    if n > MAX_N_SPLITS:
        n = MAX_N_SPLITS
    return n


@triton.jit
def _verify_prefix_stage1(
    Q,  # [extend_tokens, H_Q, D]
    K_Buffer,  # [pool_tokens, H_KV, D]
    V_Buffer,  # [pool_tokens, H_KV, Dv]
    sm_scale,
    k_scale,  # fp8 dequant multiplier for prefix K (1.0 if bf16)
    v_scale,  # fp8 dequant multiplier for prefix V (1.0 if bf16)
    qo_indptr,  # [BS+1] int32  -> rows of Q (draft queries)
    kv_indptr,  # [BS+1] int32  -> rows of kv_indices (prefix)
    kv_indices,  # [sum prefix] int64
    Att_Out,  # [BS, H_Q, N_SPLITS, L_EXT, Dv]  fp32
    Att_Lse,  # [BS, H_Q, N_SPLITS, L_EXT]      fp32
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,  # padded power-of-2 row tile (>= real l_ext)
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    MIN_SPLIT_KEYS: tl.constexpr = 0,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)

    # real number of draft query tokens for this seq
    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

    # split sizing identical to the decode kernel
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, N_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    # Short prefixes use fewer, larger splits (the rest exit at once): the split
    # count is fixed when a HIP graph is captured, the prefix length is not.
    kv_len_per_split = tl.maximum(kv_len_per_split, MIN_SPLIT_KEYS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([L_EXT], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([L_EXT], dtype=tl.float32)
    acc = tl.zeros([L_EXT, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # q tile: [L_EXT, D]
        offs_q = (
            (cur_q_start + offs_l)[:, None] * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q = tl.load(
            Q + offs_q,
            mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        q_k = q.to(K_Buffer.dtype.element_ty)

        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=n_mask,
                other=0,
            )
            # K block: [D, BLOCK_N]
            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_d[:, None] < HEAD_DIM) & n_mask[None, :],
                other=0.0,
            )
            qk = tl.dot(q_k, k)  # [L_EXT, BLOCK_N]
            qk *= sm_scale * k_scale  # fp8 dequant of prefix K (k_scale==1 if bf16)
            # NO causal mask: full prefix is visible to all draft tokens.
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # V block: [BLOCK_N, Dv]
            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=n_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # fp8 dequant of prefix V: scale the accumulated (pre-normalised) output.
        acc *= v_scale

        offs_o = (
            cur_batch * stride_ob
            + cur_head * stride_oh
            + split_kv_id * stride_os
            + offs_l[:, None] * stride_ol
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_o,
            acc / e_sum[:, None],
            mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        )

        offs_lse = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + split_kv_id * stride_ls
            + offs_l
        )
        tl.store(Att_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_l)
    else:
        # split did not run: write a sentinel lse so stage2 can ignore it.
        offs_lse = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + split_kv_id * stride_ls
            + offs_l
        )
        tl.store(
            Att_Lse + offs_lse,
            tl.zeros([L_EXT], tl.float32) - float("inf"),
            mask=mask_l,
        )


@triton.jit
def _verify_combine_stage2(
    Att_Out,  # [BS, H_Q, N_SPLITS, L_EXT, Dv]  fp32
    Att_Lse,  # [BS, H_Q, N_SPLITS, L_EXT]      fp32
    Q,  # [extend_tokens, H_Q, D]   (draft queries)
    K_Extend,  # [extend_tokens, H_KV, D]
    V_Extend,  # [extend_tokens, H_KV, Dv]
    O_Out,  # [extend_tokens, H_Q, Dv]  (final, written)
    sm_scale,
    qo_indptr,  # [BS+1] int32
    kv_indptr,  # [BS+1] int32 (prefix lengths)
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_qbs,
    stride_qh,
    stride_kebs,
    stride_keh,
    stride_vebs,
    stride_veh,
    stride_oobs,
    stride_ooh,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_CHUNK: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    MIN_SPLIT_KEYS: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext
    # The splits stage 1 actually ran (same sizing): the rest hold only -inf sentinels.
    prefix_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    kv_len_per_split = tl.maximum(
        tl.cdiv(tl.cdiv(prefix_len, N_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV, MIN_SPLIT_KEYS
    )
    n_active = tl.cdiv(prefix_len, kv_len_per_split)

    # ---- (a) combine prefix splits (logsumexp) ----------------------------
    # SPLIT_CHUNK active splits at a time with a running max, so N_SPLITS can exceed
    # what one [N_SPLITS, L_EXT, Dv] tile holds. Empty splits (LSE -inf; their partial
    # outputs are never written) get weight 0 and their outputs are not read.
    offs_s = tl.arange(0, SPLIT_CHUNK)
    m_p = tl.zeros([L_EXT], dtype=tl.float32) - 1.0e30
    denom_p = tl.zeros([L_EXT], dtype=tl.float32)
    o_prefix = tl.zeros([L_EXT, BLOCK_DV], dtype=tl.float32)
    for s0 in range(0, n_active, SPLIT_CHUNK):
        offs_lse = (
            cur_batch * stride_lb
            + cur_head * stride_lh
            + (s0 + offs_s)[:, None] * stride_ls
            + offs_l[None, :]
        )
        lse = tl.load(
            offs_lse + Att_Lse,
            mask=((s0 + offs_s) < n_active)[:, None] & mask_l[None, :],
            other=float("-inf"),
        )  # [SPLIT_CHUNK, L_EXT]
        m_new = tl.maximum(m_p, tl.max(lse, 0))
        rescale = tl.exp(m_p - m_new)
        w = tl.exp(lse - m_new[None, :])  # -inf -> 0
        # Att_Out[b,h,s,l,dv]
        offs_ao = (
            cur_batch * stride_ob
            + cur_head * stride_oh
            + (s0 + offs_s)[:, None, None] * stride_os
            + offs_l[None, :, None] * stride_ol
            + offs_dv[None, None, :]
        )
        ao = tl.load(
            offs_ao + Att_Out,
            mask=(w[:, :, None] > 0) & mask_l[None, :, None] & (offs_dv[None, None, :] < V_HEAD_DIM),
            other=0.0,
        )  # [SPLIT_CHUNK, L_EXT, Dv]
        o_prefix = o_prefix * rescale[:, None] + tl.sum(ao * w[:, :, None], 0)
        denom_p = denom_p * rescale + tl.sum(w, 0)
        m_p = m_new
    o_prefix = o_prefix / denom_p[:, None]
    lse_prefix = m_p + tl.log(denom_p)  # [L_EXT]

    # ---- (b) draft-draft causal attention (L_EXT x L_EXT) -----------------
    # load draft queries [L_EXT, D], draft K/V [L_EXT, D]/[L_EXT, Dv]
    offs_q = (
        (cur_q_start + offs_l)[:, None] * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q + offs_q,
        mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    offs_ke = (
        (cur_q_start + offs_l)[:, None] * stride_kebs
        + cur_kv_head * stride_keh
        + offs_d[None, :]
    )
    ke = tl.load(
        K_Extend + offs_ke,
        mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    offs_ve = (
        (cur_q_start + offs_l)[:, None] * stride_vebs
        + cur_kv_head * stride_veh
        + offs_dv[None, :]
    )
    ve = tl.load(
        V_Extend + offs_ve,
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    # scores[i,j] = q_i . k_j  (i query, j key)  -> [L_EXT, L_EXT]
    qk = tl.sum(q[:, None, :] * ke[None, :, :], 2) * sm_scale
    # causal among drafts: query i sees key j iff j <= i, and both valid
    causal = (offs_l[None, :] <= offs_l[:, None]) & mask_l[None, :] & mask_l[:, None]
    qk = tl.where(causal, qk, float("-inf"))
    m_d = tl.max(qk, 1)  # [L_EXT]
    pd = tl.exp(qk - m_d[:, None])  # [L_EXT, L_EXT]
    denom_d = tl.sum(pd, 1)  # [L_EXT]
    o_draft = tl.sum(pd[:, :, None] * ve[None, :, :], 1)  # [L_EXT, Dv]
    o_draft = o_draft / denom_d[:, None]
    lse_draft = m_d + tl.log(denom_d)  # [L_EXT]

    # ---- (c) final LSE merge (prefix vs draft) ----------------------------
    m = tl.maximum(lse_prefix, lse_draft)
    wp = tl.exp(lse_prefix - m)
    wd = tl.exp(lse_draft - m)
    o = (o_prefix * wp[:, None] + o_draft * wd[:, None]) / (wp + wd)[:, None]

    offs_oo = (
        (cur_q_start + offs_l)[:, None] * stride_oobs
        + cur_head * stride_ooh
        + offs_dv[None, :]
    )
    tl.store(
        O_Out + offs_oo,
        o.to(O_Out.dtype.element_ty),
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
    )


@triton.jit
def _verify_prefix_stage1_gqa(
    Q,  # [extend_tokens, H_Q, D]
    K_Buffer,  # [pool_tokens, H_KV, D]
    V_Buffer,  # [pool_tokens, H_KV, Dv]
    sm_scale,
    k_scale,
    v_scale,
    qo_indptr,
    kv_indptr,
    kv_indices,
    Att_Out,  # [BS, H_Q, N_SPLITS, L_EXT, Dv]  fp32
    Att_Lse,  # [BS, H_Q, N_SPLITS, L_EXT]      fp32
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,  # per-head padded row count (power of 2)
    ROWS: tl.constexpr,  # rows per program: a chunk of kv_group_num * L_EXT (power of 2)
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    MIN_SPLIT_KEYS: tl.constexpr = 0,
):
    """_verify_prefix_stage1 with the query heads of one KV head packed into one tile.

    Program (b, kv_head * n_chunks + chunk, split) handles rows
    [chunk * ROWS, (chunk + 1) * ROWS) of the kv_group_num * L_EXT (head, draft-token)
    rows that share kv_head, so each prefix K/V block is loaded once for all of them
    (the per-head kernel streams it kv_group_num times). Outputs and LSEs land in the
    per-head layout stage 2 reads."""
    cur_batch = tl.program_id(0)
    n_chunks: tl.constexpr = (kv_group_num * L_EXT + ROWS - 1) // ROWS
    cur_kv_head = tl.program_id(1) // n_chunks
    chunk = tl.program_id(1) % n_chunks
    split_kv_id = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    rows = chunk * ROWS + tl.arange(0, ROWS)
    g = rows // L_EXT  # query head within the group
    l = rows % L_EXT  # draft token
    head = cur_kv_head * kv_group_num + g

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_r = (g < kv_group_num) & (l < l_ext)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, N_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    # Short prefixes use fewer, larger splits (the rest exit at once): the split
    # count is fixed when a HIP graph is captured, the prefix length is not.
    kv_len_per_split = tl.maximum(kv_len_per_split, MIN_SPLIT_KEYS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    offs_lse = cur_batch * stride_lb + head * stride_lh + split_kv_id * stride_ls + l
    if split_kv_end > split_kv_start:
        q = tl.load(
            Q + (cur_q_start + l)[:, None] * stride_qbs + head[:, None] * stride_qh + offs_d[None, :],
            mask=mask_r[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        q_k = q.to(K_Buffer.dtype.element_ty)
        e_max = tl.zeros([ROWS], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([ROWS], dtype=tl.float32)
        acc = tl.zeros([ROWS, BLOCK_DV], dtype=tl.float32)
        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end
            kv_loc = tl.load(kv_indices + cur_batch_kv_start_idx + offs_n, mask=n_mask, other=0)
            k = tl.load(
                K_Buffer + kv_loc[None, :] * stride_buf_kbs + base_offs_k,
                mask=(offs_d[:, None] < HEAD_DIM) & n_mask[None, :],
                other=0.0,
            )
            qk = tl.dot(q_k, k) * (sm_scale * k_scale)
            qk = tl.where(n_mask[None, :], qk, float("-inf"))
            v = tl.load(
                V_Buffer + kv_loc[:, None] * stride_buf_vbs + base_offs_v,
                mask=n_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
                other=0.0,
            )
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
        acc *= v_scale
        tl.store(
            Att_Out
            + cur_batch * stride_ob
            + head[:, None] * stride_oh
            + split_kv_id * stride_os
            + l[:, None] * stride_ol
            + offs_dv[None, :],
            acc / e_sum[:, None],
            mask=mask_r[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        )
        tl.store(Att_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_r)
    else:
        tl.store(Att_Lse + offs_lse, tl.zeros([ROWS], tl.float32) - float("inf"), mask=mask_r)


@triton.jit
def _verify_combine_stage2_tree(
    Att_Out,  # [BS, H_Q, N_SPLITS, L_EXT, Dv]  fp32
    Att_Lse,  # [BS, H_Q, N_SPLITS, L_EXT]      fp32
    Q,  # [extend_tokens, H_Q, D]   (draft-tree queries)
    K_Extend,  # [extend_tokens, H_KV, D]
    V_Extend,  # [extend_tokens, H_KV, Dv]
    O_Out,  # [extend_tokens, H_Q, Dv]  (final, written)
    Mask,  # flattened tree mask (bool / uint8), extend_attention_fwd's layout
    sm_scale,
    qo_indptr,  # [BS+1] int32
    kv_indptr,  # [BS+1] int32 (prefix lengths)
    mask_indptr,  # [BS+1] int32 (start of each sequence's mask rows)
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_qbs,
    stride_qh,
    stride_kebs,
    stride_keh,
    stride_vebs,
    stride_veh,
    stride_oobs,
    stride_ooh,
    kv_group_num: tl.constexpr,
    N_SPLITS: tl.constexpr,
    L_EXT: tl.constexpr,  # padded power-of-2 row tile, >= 16 (tl.dot)
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    DV_TILE: tl.constexpr,  # output columns per program (grid axis 2 covers BLOCK_DV)
    MIN_BLOCK_KV: tl.constexpr,
    MIN_SPLIT_KEYS: tl.constexpr,
    SPLIT_UNROLL: tl.constexpr = 4,
):
    """_verify_combine_stage2 for a draft TREE (topk > 1): the draft-draft block
    follows the custom tree mask instead of the causal one. The mask is read
    exactly where extend_attention_fwd reads it for its extend part: row i of
    sequence b at mask_indptr[b] + i * (prefix_len + l_ext) + prefix_len + j.
    The prefix columns (all visible, skip_prefix_custom_mask) are covered by
    stage 1. The draft-draft block uses tl.dot: a tree verifies up to 64 nodes,
    where the chain kernel's broadcast product would not fit in registers.
    Each program writes DV_TILE output columns (grid axis 2), recomputing the small
    draft-draft scores: more programs and smaller split loads than one per head."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.program_id(2) * DV_TILE + tl.arange(0, DV_TILE)
    offs_l = tl.arange(0, L_EXT)
    offs_s = tl.arange(0, N_SPLITS)

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext
    prefix_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    mask_start = tl.load(mask_indptr + cur_batch)
    # The splits stage 1 actually ran (same sizing): the rest hold only -inf sentinels.
    kv_len_per_split = tl.maximum(
        tl.cdiv(tl.cdiv(prefix_len, N_SPLITS), MIN_BLOCK_KV) * MIN_BLOCK_KV, MIN_SPLIT_KEYS
    )
    n_active = tl.cdiv(prefix_len, kv_len_per_split)

    # ---- (a) combine prefix splits (logsumexp) ----------------------------
    offs_lse = (
        cur_batch * stride_lb
        + cur_head * stride_lh
        + offs_s[:, None] * stride_ls
        + offs_l[None, :]
    )
    lse = tl.load(
        offs_lse + Att_Lse,
        mask=(offs_s[:, None] < n_active) & mask_l[None, :],
        other=float("-inf"),
    )
    m_p = tl.max(lse, 0)
    m_p_safe = tl.where(m_p == float("-inf"), 0.0, m_p)
    denom_p = tl.sum(tl.exp(lse - m_p_safe[None, :]), 0)
    # Weighted sum of the split partials, SPLIT_UNROLL splits per iteration (their loads
    # in flight together): [N_SPLITS, L_EXT, Dv] in one tile does not fit in registers
    # at tree sizes (L_EXT up to 64).
    o_prefix = tl.zeros([L_EXT, DV_TILE], dtype=tl.float32)
    for s0 in range(0, n_active, SPLIT_UNROLL):
        for u in tl.static_range(SPLIT_UNROLL):
            s = s0 + u
            lse_s = tl.load(
                Att_Lse + cur_batch * stride_lb + cur_head * stride_lh + s * stride_ls + offs_l,
                mask=mask_l & (s < n_active),
                other=float("-inf"),
            )
            w_s = tl.exp(lse_s - m_p_safe)
            ao_s = tl.load(
                Att_Out
                + cur_batch * stride_ob
                + cur_head * stride_oh
                + s * stride_os
                + offs_l[:, None] * stride_ol
                + offs_dv[None, :],
                mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM) & (w_s[:, None] > 0.0),
                other=0.0,
            )
            o_prefix += ao_s * w_s[:, None]
    denom_p_safe = tl.where(denom_p == 0.0, 1.0, denom_p)
    o_prefix = o_prefix / denom_p_safe[:, None]
    lse_prefix = tl.where(denom_p == 0.0, float("-inf"), m_p_safe + tl.log(denom_p_safe))

    # ---- (b) draft-draft tree attention (L_EXT x L_EXT) -------------------
    q = tl.load(
        Q + (cur_q_start + offs_l)[:, None] * stride_qbs + cur_head * stride_qh + offs_d[None, :],
        mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    # K^T tile [D, L_EXT]
    ke_t = tl.load(
        K_Extend + (cur_q_start + offs_l)[None, :] * stride_kebs + cur_kv_head * stride_keh + offs_d[:, None],
        mask=mask_l[None, :] & (offs_d[:, None] < HEAD_DIM),
        other=0.0,
    )
    ve = tl.load(
        V_Extend + (cur_q_start + offs_l)[:, None] * stride_vebs + cur_kv_head * stride_veh + offs_dv[None, :],
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    qk = tl.dot(q, ke_t.to(q.dtype)) * sm_scale  # [L_EXT, L_EXT] fp32
    valid = mask_l[:, None] & mask_l[None, :]
    tree = tl.load(
        Mask + mask_start + offs_l[:, None] * (prefix_len + l_ext) + prefix_len + offs_l[None, :],
        mask=valid,
        other=0,
    )
    keep = valid & (tree != 0)
    qk = tl.where(keep, qk, float("-inf"))
    m_d = tl.max(qk, 1)
    m_d_safe = tl.where(m_d == float("-inf"), 0.0, m_d)
    pd = tl.exp(qk - m_d_safe[:, None])
    denom_d = tl.sum(pd, 1)
    denom_d_safe = tl.where(denom_d == 0.0, 1.0, denom_d)
    o_draft = tl.dot(pd.to(ve.dtype), ve) / denom_d_safe[:, None]  # [L_EXT, Dv]
    lse_draft = tl.where(denom_d == 0.0, float("-inf"), m_d_safe + tl.log(denom_d_safe))

    # ---- (c) final LSE merge (prefix vs draft) ----------------------------
    m = tl.maximum(lse_prefix, lse_draft)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    wp = tl.exp(lse_prefix - m_safe)
    wd = tl.exp(lse_draft - m_safe)
    total = wp + wd
    total = tl.where(total == 0.0, 1.0, total)
    o = (o_prefix * wp[:, None] + o_draft * wd[:, None]) / total[:, None]
    tl.store(
        O_Out + (cur_q_start + offs_l)[:, None] * stride_oobs + cur_head * stride_ooh + offs_dv[None, :],
        o.to(O_Out.dtype.element_ty),
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
    )


class VerifySplitKV:
    """Pre-allocates scratch buffers for a problem shape and runs the split-KV
    verify attention end to end (two Triton launches: prefix split-KV + fused
    combine/draft/merge). Buffers are sized by ``max_bs`` (constant for the
    server lifetime) and reused for every batch size <= max_bs, so their
    addresses stay fixed (CUDA/HIP-graph safe) and GPU memory does not grow per
    batch size. The kernel grid uses the actual per-call bs (<= max_bs)."""

    def __init__(
        self,
        max_bs,
        h_q,
        h_kv,
        head_dim,
        v_head_dim,
        l_ext,
        device="cuda",
        n_splits=DEFAULT_N_SPLITS,
        block_n=DEFAULT_BLOCK_N,
        num_warps=DEFAULT_NUM_WARPS,
        tree=False,
        gqa=False,
        fly=False,
    ):
        self.h_q = h_q
        self.h_kv = h_kv
        self.group = h_q // h_kv
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.l_ext = l_ext  # real draft tokens per seq (fixed == 4)
        # A tree's draft-draft block runs on tl.dot, which needs >= 16 rows.
        self.tree = tree
        self.l_pad = triton.next_power_of_2(l_ext)
        if tree:
            self.l_pad = max(16, self.l_pad)
        self.gqa = gqa
        self.fly = fly
        # Tree splits hold at least this many prefix keys (sized in-kernel from the
        # real prefix length); the prefix and combine stages must agree on it.
        if fly:
            self.min_split_keys = FLY_TREE_MIN_SPLIT_KEYS if tree else FLY_CHAIN_MIN_SPLIT_KEYS
        else:
            self.min_split_keys = TREE_MIN_SPLIT_KEYS if tree else 0
        self.device = device
        self.n_splits = n_splits
        self.block_n = block_n
        self.num_warps = num_warps
        self._alloc(max_bs)

    def _alloc(self, max_bs):
        # prefix split partials (fp32), sized for the maximum batch size.
        self.max_bs = max_bs
        self.att_out = torch.empty(
            (max_bs, self.h_q, self.n_splits, self.l_pad, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.att_lse = torch.empty(
            (max_bs, self.h_q, self.n_splits, self.l_pad),
            dtype=torch.float32,
            device=self.device,
        )

    def grow_buffers(self, max_bs):
        if max_bs > self.max_bs:
            self._alloc(max_bs)

    def _run_prefix_kernel(
        self,
        bs,
        q_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
    ):
        if self.fly:
            from sglang.kernels.ops.attention.verify_prefix_flydsl import verify_prefix_gqa

            verify_prefix_gqa(
                q_extend, k_buffer, v_buffer, kv_indptr, kv_indices, self.att_out, self.att_lse,
                bs=bs, l_ext=self.l_ext, l_pad=self.l_pad, n_splits=self.n_splits,
                min_split_keys=self.min_split_keys, sm_scale=sm_scale,
            )
            return
        if self.gqa:
            rows, n_chunks = gqa_rows(self.group, self.l_pad)
            _verify_prefix_stage1_gqa[(bs, self.h_kv * n_chunks, self.n_splits)](
                q_extend,
                k_buffer,
                v_buffer,
                sm_scale,
                k_scale,
                v_scale,
                qo_indptr,
                kv_indptr,
                kv_indices,
                self.att_out,
                self.att_lse,
                q_extend.stride(0),
                q_extend.stride(1),
                k_buffer.stride(0),
                k_buffer.stride(1),
                v_buffer.stride(0),
                v_buffer.stride(1),
                self.att_out.stride(0),
                self.att_out.stride(1),
                self.att_out.stride(2),
                self.att_out.stride(3),
                self.att_lse.stride(0),
                self.att_lse.stride(1),
                self.att_lse.stride(2),
                kv_group_num=self.group,
                N_SPLITS=self.n_splits,
                L_EXT=self.l_pad,
                ROWS=rows,
                HEAD_DIM=self.head_dim,
                V_HEAD_DIM=self.v_head_dim,
                BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
                BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
                BLOCK_N=GQA_BLOCK_N,
                MIN_BLOCK_KV=_MIN_BLOCK_KV,
                MIN_SPLIT_KEYS=self.min_split_keys,
                num_warps=GQA_NUM_WARPS,
                num_stages=1,
                **_AMD_LAUNCH_KWARGS,
            )
            return
        grid = (bs, self.h_q, self.n_splits)
        _verify_prefix_stage1[grid](
            q_extend,
            k_buffer,
            v_buffer,
            sm_scale,
            k_scale,
            v_scale,
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.att_out,
            self.att_lse,
            q_extend.stride(0),
            q_extend.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            MIN_SPLIT_KEYS=self.min_split_keys,
            num_warps=self.num_warps,
            num_stages=1,
            **_AMD_LAUNCH_KWARGS,
        )

    def _run_combine_kernel(
        self, bs, q_extend, k_extend, v_extend, o_out, qo_indptr, kv_indptr, sm_scale
    ):
        grid = (bs, self.h_q)
        _verify_combine_stage2[grid](
            self.att_out,
            self.att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            sm_scale,
            qo_indptr,
            kv_indptr,
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            o_out.stride(0),
            o_out.stride(1),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            SPLIT_CHUNK=min(self.n_splits, MAX_N_SPLITS),
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            MIN_SPLIT_KEYS=self.min_split_keys,
            num_warps=1,
            num_stages=1,
        )

    def _run_combine_tree_kernel(
        self, bs, q_extend, k_extend, v_extend, o_out, qo_indptr, kv_indptr,
        custom_mask, mask_indptr, sm_scale,
    ):
        dv_tile = min(TREE_COMBINE_DV_TILE, triton.next_power_of_2(self.v_head_dim))
        grid = (bs, self.h_q, triton.next_power_of_2(self.v_head_dim) // dv_tile)
        _verify_combine_stage2_tree[grid](
            self.att_out,
            self.att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            custom_mask,
            sm_scale,
            qo_indptr,
            kv_indptr,
            mask_indptr,
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            o_out.stride(0),
            o_out.stride(1),
            kv_group_num=self.group,
            N_SPLITS=self.n_splits,
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            DV_TILE=dv_tile,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            MIN_SPLIT_KEYS=self.min_split_keys,
            num_warps=4,
            num_stages=1,
        )

    def __call__(
        self,
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        o_out=None,
        k_scale=1.0,
        v_scale=1.0,
        custom_mask=None,
        mask_indptr=None,
    ):
        if o_out is None:
            o_out = torch.empty(
                (q_extend.shape[0], self.h_q, self.v_head_dim),
                dtype=q_extend.dtype,
                device=q_extend.device,
            )
        # actual batch size for this call (<= max_bs); the grid uses it while the
        # scratch buffers stay max_bs-sized (only the first bs slices are touched).
        bs = qo_indptr.shape[0] - 1
        # 1. prefix split-KV
        self._run_prefix_kernel(
            bs,
            q_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            k_scale,
            v_scale,
        )
        # 2+3+4. fused combine + draft-draft + merge
        if self.tree:
            self._run_combine_tree_kernel(
                bs, q_extend, k_extend, v_extend, o_out, qo_indptr, kv_indptr,
                custom_mask, mask_indptr, sm_scale,
            )
            return o_out
        self._run_combine_kernel(
            bs,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            qo_indptr,
            kv_indptr,
            sm_scale,
        )
        return o_out


# ---------------------------------------------------------------------------
# Live-server dispatch entry.
# ---------------------------------------------------------------------------
# Cache one VerifySplitKV instance per (h_q, h_kv, head_dim, v_head_dim, l_ext,
# device, n_splits) shape -- NOT keyed on the dynamic batch size. Buffers are
# sized by the stable max_bs (grown only if a larger one is ever requested), so
# a single instance serves every batch size: addresses stay fixed (graph-safe)
# and GPU memory does not grow per batch size.
_VK_CACHE = {}


def _get_vk(
    max_bs, h_q, h_kv, head_dim, v_head_dim, l_ext, device, n_splits=DEFAULT_N_SPLITS,
    tree=False, gqa=False, fly=False,
):
    key = (h_q, h_kv, head_dim, v_head_dim, l_ext, str(device), n_splits, tree, gqa, fly)
    vk = _VK_CACHE.get(key)
    if vk is None:
        block_n, num_warps = block_config(head_dim)
        vk = VerifySplitKV(
            max_bs,
            h_q,
            h_kv,
            head_dim,
            v_head_dim,
            l_ext,
            device=device,
            n_splits=n_splits,
            block_n=block_n,
            num_warps=num_warps,
            tree=tree,
            gqa=gqa,
            fly=fly,
        )
        _VK_CACHE[key] = vk
    else:
        vk.grow_buffers(max_bs)
    return vk


def can_handle(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    sliding_window_size=-1,
    sinks=None,
    logit_cap=0.0,
    xai_temperature_len=-1,
    tree_mask=False,
):
    """Return True iff the split-KV verify path can serve this exact problem
    with the same result as extend_attention_fwd. Conservative: anything not
    explicitly handled -> False -> caller falls back to the baseline.

    IMPORTANT: ``custom_mask`` is intentionally NOT inspected (its values can't
    be read inside a captured HIP graph without a host sync). The kernel always
    computes pure-causal attention, which equals the tree mask ONLY at
    speculative topk == 1. The caller therefore MUST gate enablement on topk == 1
    (TritonAttnBackend does: ``use_verify_splitkv = ... and self.topk == 1``).
    At topk > 1 the tree is not causal and this path must stay disabled."""
    # No exotic features.
    if sinks is not None:
        return False
    if sliding_window_size is not None and sliding_window_size > 0:
        return False
    if logit_cap and logit_cap > 0:
        return False
    if xai_temperature_len is not None and xai_temperature_len > 0:
        return False
    if tree_mask:
        # Draft tree (topk > 1): the combine stage reads the tree mask itself.
        if custom_mask is None or mask_indptr is None:
            return False
    elif not is_causal:
        return False
    # q layout must be [tokens, H_Q, D]; head dims handled by power-of-2 pad.
    if q_extend.dim() != 3 or k_extend.dim() != 3 or v_extend.dim() != 3:
        return False
    # GQA group must divide evenly.
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    if h_kv == 0 or h_q % h_kv != 0:
        return False
    # head dims must match buffers.
    if k_buffer.shape[1] != h_kv or v_buffer.shape[1] != h_kv:
        return False
    if q_extend.shape[2] != k_extend.shape[2]:
        return False
    if q_extend.shape[2] != k_buffer.shape[2]:
        return False
    if v_extend.shape[2] != v_buffer.shape[2]:
        return False
    # MLA (head_dim != v_head_dim, e.g. DeepSeek 576 vs 512) uses a shared
    # latent KV cache and an absorbed-attention layout the split-KV verify
    # kernel is not built for; it GPU-faults on that shape. Fall back to
    # extend_attention_fwd, which handles MLA correctly.
    if q_extend.shape[2] != v_extend.shape[2]:
        return False
    # NOTE: must NOT read any tensor *values* here (no .item()/.cpu()): the
    # target-verify step runs inside a captured CUDA/HIP graph, where a
    # device->host sync raises hipErrorStreamCaptureUnsupported. We therefore
    # gate purely on static shapes/dtypes/python scalars.
    bs = qo_indptr.shape[0] - 1
    if bs < 1:
        return False
    # max_len_extend must be a known positive python int (it is the static
    # server_args.speculative_num_draft_tokens for the verify path). For
    # topk=1 the per-seq extend len is constant == num_draft_tokens ==
    # max_len_extend by construction of qo_indptr (arange with that step), so
    # the L_EXT row-tile mask is exactly right and the tree custom_mask equals
    # causal -- no value inspection required.
    try:
        mle = int(max_len_extend)
    except (TypeError, ValueError):
        return False
    if mle < 1:
        return False
    if tree_mask and mle > 64:
        return False
    # The packed extend tensor must hold exactly bs * max_len_extend rows
    # (constant extend len). This is a pure shape check (no sync) and rejects
    # any ragged/variable-extend batch -> falls back to the baseline.
    if q_extend.shape[0] != bs * mle:
        return False
    return True


def verify_splitkv_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    max_bs=None,
    tree_mask=False,
):
    """Drop-in for extend_attention_fwd on the EAGLE target-verify shape:
    a causal chain (topk=1), or with ``tree_mask`` a draft tree (topk > 1)
    whose draft-draft block follows ``custom_mask``. Returns True if it ran (o_extend written), False if the case is
    unsupported and the caller must fall back to extend_attention_fwd.

    ``max_bs`` (optional) is the stable maximum batch size used to size the
    cached scratch buffers; the backend passes its req_to_token_pool size. If
    omitted it defaults to this call's bs.

    Arg order mirrors extend_attention_fwd exactly so the call site is a
    one-line swap.
    """
    if not can_handle(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_len_extend,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        tree_mask=tree_mask,
    ):
        return False

    bs = qo_indptr.shape[0] - 1
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    head_dim = q_extend.shape[2]
    v_head_dim = v_extend.shape[2]
    l_ext = int(max_len_extend)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    # k_scale/v_scale may be float or 0-d tensor; coerce to python float.
    try:
        k_scale = float(k_scale)
    except (TypeError, ValueError):
        k_scale = 1.0
    try:
        v_scale = float(v_scale)
    except (TypeError, ValueError):
        v_scale = 1.0

    # Adaptive split count from the average prefix length. This is a
    # pure-shape derivation (kv_indices.shape[0] / bs) -- no device->host sync,
    # so it is safe inside a captured HIP graph. The whole batch shares one
    # N_SPLITS (the grid dim must be a launch constexpr); the per-split kernel
    # logic still clamps each split's [start,end) to that seq's real length, so
    # mixed-length batches stay correct -- shorter seqs simply write fewer
    # active splits (the rest emit the -inf lse sentinel, ignored in stage2).
    avg_seqlen = kv_indices.shape[0] / max(1, bs)
    fly = use_flydsl(q_extend, k_buffer, v_buffer, kv_indptr, kv_indices, k_scale, v_scale)
    # Graphs are captured per batch size, not per context: pick the prefix stage by bs.
    use_gqa = fly or (bs >= GQA_MIN_BS and (TREE_GQA if tree_mask else CHAIN_GQA))
    if use_gqa:
        l_pad = triton.next_power_of_2(int(max_len_extend))
        if tree_mask:
            l_pad = max(16, l_pad)
        _, n_chunks = gqa_rows(h_q // h_kv, l_pad)
        n_splits = choose_n_splits_gqa(avg_seqlen, bs, h_kv, n_chunks)
        if not tree_mask and not fly:
            # Measured cap for the Triton chain stages (the combine handles more).
            n_splits = min(n_splits, MAX_N_SPLITS)
    else:
        n_splits = choose_n_splits(avg_seqlen)

    # Size scratch by the stable max_bs (backend passes req_to_token_pool size);
    # fall back to this call's bs if not provided / smaller.
    if max_bs is None or max_bs < bs:
        max_bs = bs
    vk = _get_vk(
        max_bs,
        h_q,
        h_kv,
        head_dim,
        v_head_dim,
        l_ext,
        q_extend.device,
        n_splits=n_splits,
        tree=tree_mask,
        gqa=use_gqa,
        fly=fly,
    )
    vk(
        q_extend,
        k_extend.contiguous(),
        v_extend.contiguous(),
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        o_out=o_extend,
        k_scale=k_scale,
        v_scale=v_scale,
        custom_mask=custom_mask if tree_mask else None,
        mask_indptr=mask_indptr if tree_mask else None,
    )

    return True

"""FlyDSL combine stage of the split-KV target verify (gfx950, bf16, head dim 256): drop-in for
verify_splitkv._verify_combine_stage2_tree (draft tree, custom mask) and _verify_combine_stage2 (causal chain), so
with the FlyDSL prefix stage (verify_prefix_flydsl.py) the verify attention is FlyDSL end to end.

One wave per (sequence, q head, DVT output columns); up to 16 draft tokens. In the transposed form of the prefix
stage, every lane's softmax state is its own query row i = lane % 16:

    S^T = K_ext . Q^T        16x16x32 bf16 MFMA over the head dim: lane (i, g) holds scores (i, j = 4g..4g+3)
    O^T = V_ext^T . P^T      16x16x16 bf16 MFMA: P^T is that accumulator, as is; lane holds O[i][4g..4g+3 + c0]

so neither P nor the softmax statistics move between lanes. The prefix partials are merged in the same layout
(each lane reads 4 contiguous columns of its row from every active split) and the two halves are LSE-merged.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

HEAD_DIM = 256
WAVE = 64
NEG = -1.0e30
MIN_BLOCK_KV = 32
UNROLL = 4  # prefix splits per combine step


def _fmax(a, b):
    return fx.Float32(a).maximumf(b)


@functools.lru_cache(maxsize=None)
def build(h_q, h_kv, l_pad, n_splits, min_split_keys, sm_scale, tree, dvt):
    group = h_q // h_kv
    tiles = dvt // 16
    n_kb = HEAD_DIM // 32

    def visible(ok, j, mask, row_start, i):
        # outside the kernel body: the Python branch stays a Python branch
        if tree:
            mv = fx.Int8(fx.ptr_load(mask + row_start + arith.select(ok, j, fx.Int32(0))))
            return ok & (fx.Int32(mv) != 0)
        return ok & (j <= i)

    @flyc.kernel(name=f"verify_combine_{'tree' if tree else 'chain'}_s{n_splits}_m{min_split_keys}_d{dvt}",
                 known_block_size=[WAVE, 1, 1])
    def kernel(
        att_out: fx.Pointer,  # [max_bs, H_Q, N_SPLITS, L_PAD, 256] f32
        att_lse: fx.Pointer,  # [max_bs, H_Q, N_SPLITS, L_PAD] f32 (natural log)
        q: fx.Pointer,  # [T, H_Q, 256] bf16, token stride stride_q
        k_ext: fx.Pointer,  # [T, H_KV, 256] bf16, token stride stride_ke
        v_ext: fx.Pointer,  # [T, H_KV, 256] bf16, token stride stride_ve
        o: fx.Pointer,  # [T, H_Q, 256] bf16, token stride stride_o
        mask: fx.Pointer,  # u8 tree mask (extend_attention_fwd's layout); unused for chains
        qo_indptr: fx.Pointer,  # i32
        kv_indptr: fx.Pointer,  # i32 (prefix lengths)
        mask_indptr: fx.Pointer,  # i32
        stride_q: fx.Int32,
        stride_ke: fx.Int32,
        stride_ve: fx.Int32,
        stride_o: fx.Int32,
    ):
        lane = fx.thread_idx.x
        n16, g = lane % 16, lane // 16
        b, h, dv_blk = fx.block_idx.x, fx.block_idx.y, fx.block_idx.z
        kvh = h // group
        q0 = fx.Int32(fx.ptr_load(qo_indptr + b))
        l_ext = fx.Int32(fx.ptr_load(qo_indptr + b + 1)) - q0
        prefix = fx.Int32(fx.ptr_load(kv_indptr + b + 1)) - fx.Int32(fx.ptr_load(kv_indptr + b))
        per_split = (prefix + n_splits - 1) // n_splits
        per_split = (per_split + MIN_BLOCK_KV - 1) // MIN_BLOCK_KV * MIN_BLOCK_KV
        per_split = fx.Int32(arith.select(per_split > min_split_keys, per_split, fx.Int32(min_split_keys)))
        n_active = (prefix + per_split - 1) // per_split
        i_ok = n16 < l_ext
        i_c = arith.select(i_ok, n16, fx.Int32(0))
        c0 = dv_blk * dvt

        # ---- draft-draft block: S^T on the MFMA, row i = n16, keys j = 4g + e ----
        jrow = arith.select(i_ok, n16, fx.Int32(0))  # the K_ext row this lane feeds (row n16 of S^T)
        st = fx.Vector.filled(4, 0.0, fx.Float32)
        for kb in range_constexpr(n_kb):
            ka = fx.Vector(fx.ptr_load(k_ext + (q0 + jrow) * stride_ke + kvh * HEAD_DIM + kb * 32 + g * 8,
                                       result_type=T.vec(8, T.bf16)))
            ka = fx.Vector.from_elements([fx.BFloat16(arith.select(i_ok, ka[i], fx.BFloat16(0.0)))
                                          for i in range_constexpr(8)], fx.BFloat16)
            qb = fx.Vector(fx.ptr_load(q + (q0 + i_c) * stride_q + h * HEAD_DIM + kb * 32 + g * 8,
                                       result_type=T.vec(8, T.bf16)))
            st = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [ka, qb, st]))
        mask_start = fx.Int32(fx.ptr_load(mask_indptr + b))
        x = []
        for e in range_constexpr(4):
            j = g * 4 + e
            keep = visible(i_ok & (j < l_ext), j, mask, mask_start + i_c * (prefix + l_ext) + prefix, n16)
            x.append(fx.Float32(arith.select(keep, fx.Float32(st[e]) * sm_scale, fx.Float32(NEG))))
        m_d = _fmax(_fmax(x[0], x[1]), _fmax(x[2], x[3]))
        m_d = _fmax(m_d, m_d.shuffle_xor(16, WAVE))
        m_d = _fmax(m_d, m_d.shuffle_xor(32, WAVE))
        p = [fx.Float32(arith.select(x[e] > NEG, fx.Float32(fmath.exp(x[e] - m_d)), fx.Float32(0.0)))
             for e in range_constexpr(4)]
        den_d = p[0] + p[1] + p[2] + p[3]
        den_d = den_d + den_d.shuffle_xor(16, WAVE)
        den_d = den_d + den_d.shuffle_xor(32, WAVE)
        has_d = den_d > 0.0
        den_d_safe = fx.Float32(arith.select(has_d, den_d, fx.Float32(1.0)))
        lse_d = fx.Float32(arith.select(has_d, m_d + fx.Float32(fmath.log(den_d_safe)), fx.Float32(NEG)))
        pb = fx.Vector.from_elements([p[e].to(fx.BFloat16) for e in range_constexpr(4)], fx.BFloat16).bitcast(fx.Int16)

        # ---- prefix splits, online over UNROLL splits per step (their loads in flight together): row i = n16,
        # columns c0 + 16 t + 4g .. + 3 ----
        lse_base = att_lse + (b * h_q + h) * n_splits * l_pad + i_c
        out_base = att_out + ((b * h_q + h) * n_splits * l_pad + i_c) * HEAD_DIM + c0 + g * 4
        m_p = fx.Float32(NEG)
        den_p = fx.Float32(0.0)
        acc = fx.Vector.filled(4 * tiles, 0.0, fx.Float32)
        for s0 in range(0, n_active, UNROLL):
            ss = [fx.Int32(arith.select(s0 + u < n_active, s0 + u, n_active - 1)) for u in range_constexpr(UNROLL)]
            lse_u = [fx.Float32(arith.select(s0 + u < n_active, fx.Float32(fx.ptr_load(lse_base + ss[u] * l_pad)),
                                             fx.Float32(NEG))) for u in range_constexpr(UNROLL)]
            aos = [[fx.Vector(fx.ptr_load(out_base + ss[u] * l_pad * HEAD_DIM + t * 16, result_type=T.vec(4, T.f32)))
                    for t in range_constexpr(tiles)] for u in range_constexpr(UNROLL)]
            m_new = m_p
            for u in range_constexpr(UNROLL):
                m_new = _fmax(m_new, lse_u[u])
            corr = fx.Float32(fmath.exp(m_p - m_new))
            w = [fx.Float32(arith.select(lse_u[u] > NEG, fx.Float32(fmath.exp(lse_u[u] - m_new)), fx.Float32(0.0)))
                 for u in range_constexpr(UNROLL)]
            den_new = den_p * corr
            for u in range_constexpr(UNROLL):
                den_new = den_new + w[u]
            parts = []
            for t in range_constexpr(tiles):
                for e in range_constexpr(4):
                    val = acc[t * 4 + e] * corr
                    for u in range_constexpr(UNROLL):
                        val = val + w[u] * fx.Float32(aos[u][t][e])
                    parts.append(val)
            acc = fx.Vector.from_elements(parts, fx.Float32)
            den_p = den_new
            m_p = m_new
        has_p = den_p > 0.0
        den_p_safe = fx.Float32(arith.select(has_p, den_p, fx.Float32(1.0)))
        lse_p = fx.Float32(arith.select(has_p, m_p + fx.Float32(fmath.log(den_p_safe)), fx.Float32(NEG)))

        # ---- LSE merge ----
        mm_ = _fmax(lse_p, lse_d)
        wp = fx.Float32(arith.select(has_p, fx.Float32(fmath.exp(lse_p - mm_)), fx.Float32(0.0)))
        wd = fx.Float32(arith.select(has_d, fx.Float32(fmath.exp(lse_d - mm_)), fx.Float32(0.0)))
        tot = wp + wd
        tot = fx.Float32(arith.select(tot > 0.0, tot, fx.Float32(1.0)))
        cp = wp / (den_p_safe * tot)
        cd = wd / (den_d_safe * tot)
        for t in range_constexpr(tiles):
            # V_ext^T operand: lane (dv = n16, g) holds V[4g .. 4g + 3][c0 + 16 t + n16]
            col = c0 + t * 16 + n16
            vv = []
            for e in range_constexpr(4):
                j = g * 4 + e
                vj = fx.BFloat16(fx.ptr_load(v_ext + (q0 + arith.select(j < l_ext, j, fx.Int32(0))) * stride_ve
                                             + kvh * HEAD_DIM + col))
                vv.append(fx.BFloat16(arith.select(j < l_ext, vj, fx.BFloat16(0.0))))
            # (the 1k bf16 MFMA takes its operands as i16 vectors)
            od = fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [
                fx.Vector.from_elements(vv, fx.BFloat16).bitcast(fx.Int16), pb,
                fx.Vector.filled(4, 0.0, fx.Float32)]))
            res = [(acc[t * 4 + e] * cp + fx.Float32(od[e]) * cd).to(fx.BFloat16) for e in range_constexpr(4)]
            if i_ok:
                fx.ptr_store(fx.Vector.from_elements(res, fx.BFloat16),
                             o + (q0 + n16) * stride_o + h * HEAD_DIM + c0 + t * 16 + g * 4)

    @flyc.jit
    def launch(att_out: fx.Pointer, att_lse: fx.Pointer, q: fx.Pointer, k_ext: fx.Pointer, v_ext: fx.Pointer,
               o: fx.Pointer, mask: fx.Pointer, qo_indptr: fx.Pointer, kv_indptr: fx.Pointer,
               mask_indptr: fx.Pointer, stride_q: fx.Int32, stride_ke: fx.Int32, stride_ve: fx.Int32,
               stride_o: fx.Int32, bs: fx.Int32, stream: fx.Stream):
        kernel(att_out, att_lse, q, k_ext, v_ext, o, mask, qo_indptr, kv_indptr, mask_indptr, stride_q, stride_ke,
               stride_ve, stride_o).launch(grid=(bs, h_q, HEAD_DIM // dvt), block=(WAVE, 1, 1), stream=stream)

    return launch


def _ptr(t, ty):
    return flyc.from_c_void_p(ty, t.data_ptr(), assumed_align=16 if t.data_ptr() % 16 == 0 else 1)


def _int_ty(t):
    import torch

    return {torch.int64: fx.Int64, torch.int32: fx.Int32, torch.bool: fx.Int8, torch.uint8: fx.Int8,
            torch.int8: fx.Int8}[t.dtype]


def verify_combine(att_out, att_lse, q, k_ext, v_ext, o, qo_indptr, kv_indptr, *, bs, l_pad, n_splits,
                   min_split_keys, sm_scale, custom_mask=None, mask_indptr=None, dvt=64, stream=None):
    """o = the verify attention of every draft token: prefix partials (att_out / att_lse, from the prefix stage)
    merged with the draft-draft block (tree: custom_mask; chain: causal). Up to 16 draft tokens per sequence."""
    import torch

    tree = custom_mask is not None
    h_q, h_kv = q.shape[1], k_ext.shape[1]
    launch = build(h_q, h_kv, l_pad, n_splits, min_split_keys, float(sm_scale), tree, dvt)
    if not tree:
        custom_mask, mask_indptr = qo_indptr, qo_indptr  # unread
    s = stream if stream is not None else torch.cuda.current_stream()
    # index tensors keep their dtype (serving passes int64 for some): the kernel loads them as such
    launch(_ptr(att_out, fx.Float32), _ptr(att_lse, fx.Float32), _ptr(q, fx.BFloat16), _ptr(k_ext, fx.BFloat16),
           _ptr(v_ext, fx.BFloat16), _ptr(o, fx.BFloat16), _ptr(custom_mask, _int_ty(custom_mask)),
           _ptr(qo_indptr, _int_ty(qo_indptr)), _ptr(kv_indptr, _int_ty(kv_indptr)),
           _ptr(mask_indptr, _int_ty(mask_indptr)), q.stride(0), k_ext.stride(0), v_ext.stride(0), o.stride(0), bs,
           fx.Stream(s.cuda_stream))

"""FlyDSL prefix stage of the split-KV target verify (gfx950, bf16, head dim 256).

Drop-in for verify_splitkv._verify_prefix_stage1_gqa: program (batch, kv head * row
chunk, split) runs a flash-style online softmax over its split of the prefix for every
query row (group heads x draft tokens) sharing the kv head, and writes the per-head
partial output / LSE (natural log) the Triton combine stages read. Splits are sized
exactly as the Triton stage sizes them.

A wave owns MT x 16 query rows; the waves of a CTA share each K/V tile staged in LDS.
With MFMA 16x16x32 bf16 the roles are swapped against the usual Q K^T:

    S^T  = K . Q^T     A = K tile (keys x head dim, LDS), B = Q^T (registers)
    O^T += V^T . P^T   A = V^T (transposing LDS read), B = P^T (the S^T accumulator)

so P never round-trips through LDS, each lane's softmax state is its own query row,
and a wave's MT row tiles reuse every K / V operand it reads. Of each 32 keys a lane
holds {4g..4g+3, 16+4g..16+4g+3} (g = lane / 16): the S^T accumulator's row order,
used as the PV reduction order on both operands.

The K/V gather is software-pipelined: tile t + 1's rows and tile t + 2's token slots
are in flight while tile t computes.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

HEAD_DIM = 256
WAVE = 64
MAX_WAVES = 8
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
NEG = -1.0e30

# Tuned on MI355X (tests/test_fly_prefix.py in the bring-up workspace): 64-key tiles
# for row-heavy trees (more K/V bytes in flight per CTA), 32 for chains.
KT = 32  # keys per tile
KT_MANY_ROWS = 64  # above 32 rows per kv head
K_PITCH = HEAD_DIM + 8  # LDS row pitches in bf16; pads chosen against bank conflicts
V_PITCH = HEAD_DIM + 16
MT = 1  # 16-row tiles per wave


def _cdiv(a, b):
    return (a + b - 1) // b


def _imax(a, b):
    return fx.Int32(arith.select(a > b, a, b))


def _imin(a, b):
    return fx.Int32(arith.select(a < b, a, b))


def _fmax(a, b):
    return fx.Float32(a).maximumf(b)


def layout(group, l_ext, mt=MT):
    """(waves per CTA, row chunks per kv head) for group * l_ext rows."""
    m_tiles = _cdiv(group * l_ext, 16)
    waves = min(MAX_WAVES, _cdiv(m_tiles, mt))
    return waves, _cdiv(m_tiles, waves * mt)


@functools.lru_cache(maxsize=None)
def build(group, h_kv, l_ext, l_pad, n_splits, min_split_keys, sm_scale,
          kt=KT, k_pitch=K_PITCH, v_pitch=V_PITCH, mt=MT):
    rows_total = group * l_ext
    waves, n_chunks = layout(group, l_ext, mt)
    threads = waves * WAVE
    h_q = group * h_kv
    pieces_per_tile = kt * HEAD_DIM // 8  # 16 B pieces of one K (or V) tile
    load_iters = _cdiv(pieces_per_tile, threads)
    n_sub = kt // 16  # 16-key S^T tiles per K/V tile
    scale_log2 = float(sm_scale) * LOG2E
    assert kt % 32 == 0

    @fx.struct
    class Smem:
        words: fx.Array[fx.BFloat16, kt * (k_pitch + v_pitch), 16]

    @flyc.kernel(
        name=f"verify_prefix_gqa_g{group}_l{l_ext}_s{n_splits}_m{min_split_keys}_k{kt}_mt{mt}",
        known_block_size=[threads, 1, 1],
    )
    def kernel(
        q: fx.Pointer,  # [T, H_Q, 256] bf16, row stride stride_q
        k_buf: fx.Pointer,  # [pool, H_KV, 256] bf16, token stride stride_kv
        v_buf: fx.Pointer,
        kv_indptr: fx.Pointer,  # [bs + 1] i32
        kv_indices: fx.Pointer,  # i64
        att_out: fx.Pointer,  # [max_bs, H_Q, N_SPLITS, L_PAD, 256] f32
        att_lse: fx.Pointer,  # [max_bs, H_Q, N_SPLITS, L_PAD] f32
        stride_q: fx.Int32,
        stride_kv: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane, wave = tid % WAVE, tid // WAVE
        n16, g = lane % 16, lane // 16
        b, kvh_chunk, split = fx.block_idx.x, fx.block_idx.y, fx.block_idx.z
        kvh, chunk = kvh_chunk // n_chunks, kvh_chunk % n_chunks
        ktile = fx.SharedAllocator().allocate(Smem).peek().words.ptr
        vtile = ktile + kt * k_pitch
        tr_atom = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans16_64b(), fx.BFloat16)

        def tr_read(ptr):
            # ds_read_b64_tr_b16: source lane s of a 16-lane group points at 4 values of
            # row s / 4 at column 4 * (s % 4); lane t receives column t of all 4 rows.
            dst = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
            fx.copy(tr_atom, fx.make_view(ptr, fx.make_layout(4, 1)), dst)
            return fx.Vector(dst.load())

        kv_start = fx.Int32(fx.ptr_load(kv_indptr + b))
        seq_len = fx.Int32(fx.ptr_load(kv_indptr + b + 1)) - kv_start
        per_split = _imax(_cdiv(_cdiv(seq_len, n_splits), 32) * 32, fx.Int32(min_split_keys))
        start = per_split * split
        end = _imin(start + per_split, seq_len)

        # This lane's query row in each of its row tiles (a column of S^T / O^T). Rows
        # past group * l_ext are padding: computed on the last row's query, never stored.
        rows = [(chunk * waves + wave) * (16 * mt) + m * 16 + n16 for m in range_constexpr(mt)]
        rows_c = [_imin(r, fx.Int32(rows_total - 1)) for r in rows]
        heads = [kvh * group + r // l_ext for r in rows_c]
        toks = [r % l_ext for r in rows_c]
        lse_offs = [((b * h_q + heads[m]) * n_splits + split) * l_pad + toks[m] for m in range_constexpr(mt)]

        # This thread's share of a K/V tile: pass `it` stages piece tid + it * threads
        # (the last pass is partial when the threads don't divide the tile).
        pieces = [_imin(tid + it * threads, fx.Int32(pieces_per_tile - 1)) for it in range_constexpr(load_iters)]
        keys = [piece // 32 for piece in pieces]
        cols = [(piece % 32) * 8 for piece in pieces]

        def load_locs(tile):
            # Token slots of this thread's keys; keys past the prefix repeat its last slot
            # (finite data, masked out of the softmax).
            return fx.Vector.from_elements(
                [fx.Int32(fx.ptr_load(kv_indices + kv_start + _imin(tile + keys[it], seq_len - 1)))
                 for it in range_constexpr(load_iters)],
                fx.Int32,
            )

        def gather(locs):
            # K and V pieces of one tile, packed [k0, v0, k1, v1, ...] (8 bf16 each).
            elems = []
            for it in range_constexpr(load_iters):
                off = locs[it] * stride_kv + kvh * HEAD_DIM + cols[it]
                for buf in (k_buf, v_buf):
                    piece = fx.Vector(fx.ptr_load(buf + off, result_type=T.vec(8, T.bf16)))
                    elems += [piece[i] for i in range_constexpr(8)]
            return fx.Vector.from_elements(elems, fx.BFloat16)

        def mfma(a, b_, c):
            return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b_, c]))

        if end > start:
            qv = []
            for m in range_constexpr(mt):
                q_row = q + (b * l_ext + toks[m]) * stride_q + heads[m] * HEAD_DIM + g * 8
                qv.append([fx.Vector(fx.ptr_load(q_row + kb * 32, result_type=T.vec(8, T.bf16)))
                           for kb in range_constexpr(HEAD_DIM // 32)])
            # Loop-carried: running max (log2 domain) and per-lane partial row sum per
            # row tile, O^T accumulators (row tile major), the prefetched tile.
            m_run = fx.Vector.filled(mt, NEG, fx.Float32)
            l_run = fx.Vector.filled(mt, 0.0, fx.Float32)
            acc = fx.Vector.filled(64 * mt, 0.0, fx.Float32)
            kvv = gather(load_locs(start))
            locs = load_locs(start + kt)
            for tile in range(start, end, kt):
                for it in range_constexpr(load_iters):
                    for half, dst, pitch in ((0, ktile, k_pitch), (1, vtile, v_pitch)):
                        base = (2 * it + half) * 8
                        fx.ptr_store(fx.Vector.from_elements([kvv[base + i] for i in range_constexpr(8)], fx.BFloat16),
                                     dst + keys[it] * pitch + cols[it])
                gpu.barrier()
                kvv_next = gather(locs)
                locs_next = load_locs(tile + 2 * kt)

                # S^T, one accumulator per (row tile, 16-key sub-tile).
                s = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(n_sub)] for _ in range_constexpr(mt)]
                for kb in range_constexpr(HEAD_DIM // 32):
                    for h in range_constexpr(n_sub):
                        a = fx.Vector(fx.ptr_load(ktile + (h * 16 + n16) * k_pitch + kb * 32 + g * 8,
                                                  result_type=T.vec(8, T.bf16)))
                        for m in range_constexpr(mt):
                            s[m][h] = mfma(a, qv[m][kb], s[m][h])

                # Online softmax in the log2 domain; keys past `end` masked.
                valid = [tile + h * 16 + g * 4 + e < end for h in range_constexpr(n_sub) for e in range_constexpr(4)]
                next_m, next_l, corr, pb = [], [], [], []
                for m in range_constexpr(mt):
                    x = [fx.Float32(arith.select(valid[h * 4 + e], fx.Float32(s[m][h][e]) * fx.Float32(scale_log2),
                                                 fx.Float32(NEG)))
                         for h in range_constexpr(n_sub) for e in range_constexpr(4)]
                    tmax = x[0]
                    for i in range_constexpr(1, 4 * n_sub):
                        tmax = _fmax(tmax, x[i])
                    tmax = _fmax(tmax, tmax.shuffle_xor(16, 64))
                    tmax = _fmax(tmax, tmax.shuffle_xor(32, 64))
                    m_new = _fmax(m_run[m], tmax)
                    c = fx.Float32(fmath.exp2(m_run[m] - m_new))
                    p = [fx.Float32(arith.select(valid[i], fx.Float32(fmath.exp2(x[i] - m_new)), fx.Float32(0.0)))
                         for i in range_constexpr(4 * n_sub)]
                    psum = p[0]
                    for i in range_constexpr(1, 4 * n_sub):
                        psum = psum + p[i]
                    next_m.append(m_new)
                    next_l.append(l_run[m] * c + psum)
                    corr.append(c)
                    # PV operand per 32 keys: sub-tiles 2j and 2j + 1.
                    pb.append([fx.Vector.from_elements([p[(2 * j + hh) * 4 + e].to(fx.BFloat16)
                                                        for hh in range_constexpr(2) for e in range_constexpr(4)],
                                                       fx.BFloat16)
                               for j in range_constexpr(n_sub // 2)])

                # O^T += V^T . P^T over 16 head-dim tiles.
                next_acc = [[] for _ in range_constexpr(mt)]
                for nt in range_constexpr(HEAD_DIM // 16):
                    src = vtile + (g * 4 + n16 // 4) * v_pitch + nt * 16 + (n16 % 4) * 4
                    va = [tr_read(src + j * 32 * v_pitch).shuffle(tr_read(src + (j * 32 + 16) * v_pitch),
                                                                  list(range(8)))
                          for j in range_constexpr(n_sub // 2)]
                    for m in range_constexpr(mt):
                        cc = fx.Vector.from_elements([acc[(m * 16 + nt) * 4 + e] * corr[m] for e in range_constexpr(4)],
                                                     fx.Float32)
                        for j in range_constexpr(n_sub // 2):
                            cc = mfma(va[j], pb[m][j], cc)
                        next_acc[m] += [cc[e] for e in range_constexpr(4)]
                acc = fx.Vector.from_elements([v for m in range_constexpr(mt) for v in next_acc[m]], fx.Float32)
                m_run = fx.Vector.from_elements(next_m, fx.Float32)
                l_run = fx.Vector.from_elements(next_l, fx.Float32)
                kvv = kvv_next
                locs = locs_next
                # Every reader retires before the next tile overwrites LDS.
                gpu.barrier()

            for m in range_constexpr(mt):
                l_row = l_run[m] + l_run[m].shuffle_xor(16, 64)
                l_row = l_row + l_row.shuffle_xor(32, 64)
                inv = fx.Float32(1.0) / l_row
                if rows[m] < rows_total:
                    out_row = att_out + lse_offs[m] * HEAD_DIM + g * 4
                    for nt in range_constexpr(HEAD_DIM // 16):
                        fx.ptr_store(
                            fx.Vector.from_elements([acc[(m * 16 + nt) * 4 + e] * inv for e in range_constexpr(4)],
                                                    fx.Float32),
                            out_row + nt * 16,
                        )
                    if g == 0:
                        fx.ptr_store((m_run[m] + fx.Float32(fmath.log2(l_row))) * fx.Float32(LN2), att_lse + lse_offs[m])
        else:
            for m in range_constexpr(mt):
                if rows[m] < rows_total:
                    if g == 0:
                        fx.ptr_store(fx.Float32(-float("inf")), att_lse + lse_offs[m])

    @flyc.jit
    def launch(
        q: fx.Pointer,
        k_buf: fx.Pointer,
        v_buf: fx.Pointer,
        kv_indptr: fx.Pointer,
        kv_indices: fx.Pointer,
        att_out: fx.Pointer,
        att_lse: fx.Pointer,
        stride_q: fx.Int32,
        stride_kv: fx.Int32,
        bs: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(q, k_buf, v_buf, kv_indptr, kv_indices, att_out, att_lse, stride_q, stride_kv).launch(
            grid=(bs, h_kv * n_chunks, n_splits), block=(threads, 1, 1), stream=stream
        )

    return launch


def _ptr(t, ty):
    return flyc.from_c_void_p(ty, t.data_ptr(), assumed_align=16)


def verify_prefix_gqa(
    q, k_buffer, v_buffer, kv_indptr, kv_indices, att_out, att_lse, *,
    bs, l_ext, l_pad, n_splits, min_split_keys, sm_scale, stream=None, **tuning,
):
    """Fill att_out / att_lse for the prefix exactly as _verify_prefix_stage1_gqa does
    (k_scale = v_scale = 1). Shapes: q [bs * l_ext, H_Q, 256], k/v_buffer
    [pool, H_KV, 256] bf16 (head stride 256), kv_indptr i32, kv_indices i64."""
    import torch

    h_q, h_kv = q.shape[1], k_buffer.shape[1]
    group = h_q // h_kv
    tuning.setdefault("kt", KT_MANY_ROWS if group * l_ext > 32 else KT)
    launch = build(group, h_kv, l_ext, l_pad, n_splits, min_split_keys, float(sm_scale), **tuning)
    s = stream if stream is not None else torch.cuda.current_stream()
    launch(
        _ptr(q, fx.BFloat16), _ptr(k_buffer, fx.BFloat16), _ptr(v_buffer, fx.BFloat16),
        _ptr(kv_indptr, fx.Int32), _ptr(kv_indices, fx.Int64),
        _ptr(att_out, fx.Float32), _ptr(att_lse, fx.Float32),
        q.stride(0), k_buffer.stride(0), bs, fx.Stream(s.cuda_stream),
    )

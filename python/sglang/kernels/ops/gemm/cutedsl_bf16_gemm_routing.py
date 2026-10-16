"""Routing of bf16 GEMMs between the CuTe DSL TGV kernel and cuBLAS.

Plain Python, importable without the CuTe DSL, so the policy is unit-tested on
CPU; ``cutedsl_bf16_gemm`` re-exports it for its callers.
"""

# MuseSpark TP4/TP8 projections using TGV on GB300; other M use cuBLAS.
_MUSE_SPARK_TGV_M_RANGES = {
    (4096, 8192): (1, 16),  # replicated pooler
    (8192, 4096): (1, 128),  # replicated unpooler
    (4608, 8192): (1, 256),  # TP4 qkvg
    (8192, 2048): (1, 128),  # TP4 o_proj
    (1024, 8192): (1, 256),  # TP4 pooler slice
    (2048, 4096): (1, 256),  # TP4 unpooler slice
    (2304, 8192): (40, 512),  # TP8 qkvg
    (8192, 1024): (1, 64),  # TP8 o_proj
    (512, 8192): (256, 512),  # TP8 pooler slice
    (1024, 4096): (1, 512),  # TP8 unpooler slice
}


def use_cutedsl_bf16_gemm(m: int, n: int, k: int) -> bool:
    """TGV-vs-cuBLAS (``F.linear``) decision, measured on B300 under CUDA graph
    capture (cold L2). Decode batches (``m <= 16``, ``n >= 1024``) run TGV; for
    larger batches, ties and unmeasured regions fall back to cuBLAS."""
    if m <= 0:
        # Empty batch: DP-attention idle groups run a 0-token dummy forward to
        # keep the mlp-sync lockstep. A 0-CTA TGV grid is a driver-level
        # CUDA_ERROR_INVALID_VALUE, so leave empty inputs to cuBLAS.
        return False
    if k % 8 != 0:  # TMA requires 16B-aligned rows
        return False
    m_range = _MUSE_SPARK_TGV_M_RANGES.get((n, k))
    if m_range is not None:
        return m_range[0] <= m <= m_range[1]
    if m <= 16 and n >= 1024:
        # Decode batches only stream the weight, and TGV starts streaming it
        # while it waits on the previous kernel (PDL): it beat cuBLAS on every
        # dense decode shape swept (k 512-28672, n 1024-57344), also outside
        # the k / n bounds below. Narrower n leaves TGV too few CTAs.
        return True
    if n < 1024 or k < 2048 or k > 6144:
        return False
    ragged = m % 16 != 0
    if n <= 1024:
        return m <= 512 and (m >= 48 or m % 16 >= 9)
    if n <= 2624:
        return m <= 128 and k >= 4096
    if n <= 4096:
        return m <= 64
    if k < 4096:
        return k >= 3072 and 25 <= m <= 48 and ragged
    if n <= 6144:
        if m <= 64:
            return m <= 32 or k >= 6144 or ragged
        return m <= 72 and k >= 6144 and ragged
    if n <= 8192:
        return m <= 48 or (m <= 63 and ragged)
    return n <= 12288 and k >= 6144 and (m <= 32 or (m <= 48 and ragged))

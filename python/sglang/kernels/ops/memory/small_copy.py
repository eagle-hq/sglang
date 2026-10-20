import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["sizes", "cols", "source_strides", "dest_strides"],
    do_not_specialize_on_alignment=["sizes", "cols", "source_strides", "dest_strides"],
)
def _small_copy_kernel(
    sources,
    destinations,
    sizes,
    cols,
    source_strides,
    dest_strides,
    BIT_WIDTHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    field = tl.program_id(0)
    offsets = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    for i in tl.static_range(len(BIT_WIDTHS)):
        if field == i:
            src_offset = (
                offsets // cols[i] * source_strides[i][0]
                + offsets % cols[i] * source_strides[i][1]
            )
            dst_offset = (
                offsets // cols[i] * dest_strides[i][0]
                + offsets % cols[i] * dest_strides[i][1]
            )
            src = sources[i]
            dst = destinations[i]
            if BIT_WIDTHS[i] == 8:
                src = src.to(tl.pointer_type(tl.uint8))
                dst = dst.to(tl.pointer_type(tl.uint8))
            elif BIT_WIDTHS[i] == 16:
                src = src.to(tl.pointer_type(tl.uint16))
                dst = dst.to(tl.pointer_type(tl.uint16))
            elif BIT_WIDTHS[i] == 32:
                src = src.to(tl.pointer_type(tl.uint32))
                dst = dst.to(tl.pointer_type(tl.uint32))
            elif BIT_WIDTHS[i] == 64:
                src = src.to(tl.pointer_type(tl.uint64))
                dst = dst.to(tl.pointer_type(tl.uint64))
            values = tl.load(src + src_offset, offsets < sizes[i], other=0)
            tl.store(dst + dst_offset, values, offsets < sizes[i])


_SUPPORTED_DTYPES = frozenset(
    (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
)
_INDEX_DTYPES = frozenset((torch.int32, torch.int64))


def _byte_range(tensor, n, row_stride, col_stride, cols):
    # Bytes the 1-D or 2-D tensor spans; strides as passed to the kernel.
    first = tensor.data_ptr()
    if not n:
        return first, first
    rows = n // cols
    extent = (rows - 1) * row_stride + (cols - 1) * col_stride + 1
    return first, first + extent * tensor.element_size()


def try_small_copy(dsts, srcs):
    # Runs on every CUDA graph replay, so the checks stay on cheap tensor
    # attributes and read each one once.
    if len(dsts) < 2 or len(dsts) != len(srcs):
        return False
    device = dsts[0].device
    if device.type != "cuda":
        return False
    sizes, cols, src_strides, dst_strides, bit_widths = [], [], [], [], []
    src_ranges, dst_ranges = [], []
    for dst, src in zip(dsts, srcs):
        dst_dtype = dst.dtype
        src_dtype = src.dtype
        shape = dst.shape
        ndim = len(shape)
        if (
            dst_dtype not in _SUPPORTED_DTYPES
            or src_dtype not in _SUPPORTED_DTYPES
            or dst.device != device
            or src.device != device
            or src.shape != shape
            or ndim not in (1, 2)
            or dst.requires_grad
            or src.requires_grad
            or src.is_neg()
            or dst.is_neg()
        ):
            return False
        n = dst.numel()
        if n > 8192:
            return False
        same_dtype = dst_dtype == src_dtype
        if not same_dtype and not (
            dst_dtype in _INDEX_DTYPES and src_dtype in _INDEX_DTYPES
        ):
            return False
        c = max(shape[-1], 1)
        ds = dst.stride() if ndim == 2 else (0, dst.stride(0))
        ss = src.stride() if ndim == 2 else (0, src.stride(0))
        if ds[1] <= 0 or (ndim == 2 and ds[0] < c * ds[1]):
            return False
        sizes.append(n)
        cols.append(c)
        src_strides.append(ss)
        dst_strides.append(ds)
        bit_widths.append(dst.element_size() * 8 if same_dtype else 0)
        src_ranges.append(_byte_range(src, n, ss[0], ss[1], c))
        dst_ranges.append(_byte_range(dst, n, ds[0], ds[1], c))
    for i, (first, last) in enumerate(dst_ranges):
        for j, (other_first, other_last) in enumerate(src_ranges):
            if first < other_last and other_first < last:
                if (
                    i != j
                    or dsts[i].data_ptr() != srcs[i].data_ptr()
                    or dst_strides[i] != src_strides[i]
                    or dsts[i].dtype != srcs[i].dtype
                ):
                    return False
        for other_first, other_last in dst_ranges[:i]:
            if first < other_last and other_first < last:
                return False
    largest = max(sizes)
    if largest:
        _small_copy_kernel[(len(dsts), triton.cdiv(largest, 256))](
            tuple(srcs),
            tuple(dsts),
            tuple(sizes),
            tuple(cols),
            tuple(src_strides),
            tuple(dst_strides),
            tuple(bit_widths),
            BLOCK=256,
        )
    return True

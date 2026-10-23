import torch
import triton
import triton.language as tl

# Triton specializes the ints and pointers inside a tuple argument on their
# values (equal to 1, divisible by 16, 16-byte aligned) whatever
# do_not_specialize says. These copies run before every CUDA graph replay with
# sizes that follow the batch and sources wherever the allocator put them, so a
# cold server kept compiling new kernel variants on the scheduler while it
# served. Every runtime value is passed as 2 * value + 3 instead, which is odd
# and above 1, and decoded in the kernel: one compiled kernel per field layout.


def _encode(value):
    return 2 * value + 3


@triton.jit
def _decode(encoded):
    return (encoded - 3) >> 1


@triton.jit
def _pointer(encoded, BITS: tl.constexpr):
    address = _decode(encoded)
    if BITS == 8:
        return address.to(tl.pointer_type(tl.int8))
    elif BITS == 16:
        return address.to(tl.pointer_type(tl.int16))
    elif BITS == 32:
        return address.to(tl.pointer_type(tl.int32))
    else:
        return address.to(tl.pointer_type(tl.int64))


@triton.jit
def _small_copy_kernel(
    sources,
    destinations,
    sizes,
    cols,
    source_strides,
    dest_strides,
    SOURCE_BITS: tl.constexpr,
    DEST_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    field = tl.program_id(0)
    offsets = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    for i in tl.static_range(len(SOURCE_BITS)):
        if field == i:
            c = _decode(cols[i])
            row = offsets // c
            col = offsets % c
            src_offset = row * _decode(source_strides[i][0]) + col * _decode(
                source_strides[i][1]
            )
            dst_offset = row * _decode(dest_strides[i][0]) + col * _decode(
                dest_strides[i][1]
            )
            # Same widths copy the bits; int32 <-> int64 index copies convert.
            src = _pointer(sources[i], SOURCE_BITS[i])
            dst = _pointer(destinations[i], DEST_BITS[i])
            mask = offsets < _decode(sizes[i])
            values = tl.load(src + src_offset, mask, other=0)
            tl.store(dst + dst_offset, values, mask)


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


def _byte_range(first, element_size, n, row_stride, col_stride, cols):
    # Bytes the 1-D or 2-D tensor spans; strides as passed to the kernel.
    if not n:
        return first, first
    rows = n // cols
    extent = (rows - 1) * row_stride + (cols - 1) * col_stride + 1
    return first, first + extent * element_size


def try_small_copy(dsts, srcs):
    # Runs on every CUDA graph replay, so the checks stay on cheap tensor
    # attributes and read each one once.
    if len(dsts) < 2 or len(dsts) != len(srcs):
        return False
    device = dsts[0].device
    if device.type != "cuda":
        return False
    sizes, cols, src_strides, dst_strides = [], [], [], []
    src_bits, dst_bits, src_ranges, dst_ranges = [], [], [], []
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
        src_size = src.element_size()
        dst_size = dst.element_size()
        src_bits.append(src_size * 8)
        dst_bits.append(dst_size * 8)
        src_ranges.append(_byte_range(src.data_ptr(), src_size, n, ss[0], ss[1], c))
        dst_ranges.append(_byte_range(dst.data_ptr(), dst_size, n, ds[0], ds[1], c))
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
            tuple(_encode(first) for first, _ in src_ranges),
            tuple(_encode(first) for first, _ in dst_ranges),
            tuple(_encode(n) for n in sizes),
            tuple(_encode(c) for c in cols),
            tuple((_encode(row), _encode(col)) for row, col in src_strides),
            tuple((_encode(row), _encode(col)) for row, col in dst_strides),
            tuple(src_bits),
            tuple(dst_bits),
            BLOCK=256,
        )
    return True

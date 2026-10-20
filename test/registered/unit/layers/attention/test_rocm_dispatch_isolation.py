"""AMD attention additions leave NVIDIA dispatch and ratio routing unchanged."""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


@pytest.mark.parametrize(
    "function", ["flash_attn_with_kvcache", "flash_attn_varlen_func"]
)
def test_flash_attention_cuda_keeps_fa3(function):
    from sglang.kernels.ops.attention import flash_attention as module

    args = [torch.zeros(1, 1, 1, 16)] * 3
    if function == "flash_attn_varlen_func":
        args += [torch.tensor([0, 1], dtype=torch.int32)] * 2
    sentinel = object()
    with (
        patch.object(torch.version, "hip", None),
        patch.object(module, "fa3_" + function, return_value=sentinel) as native,
        patch.object(module._triton_fa, function) as rocm,
    ):
        assert getattr(module, function)(*args) is sentinel
        native.assert_called_once()
        rocm.assert_not_called()


@pytest.mark.parametrize(
    "function", ["flash_attn_with_kvcache", "flash_attn_varlen_func"]
)
def test_flash_attention_hip_rejects_cuda_only_mask_mod(function):
    from sglang.kernels.ops.attention import flash_attention as module

    args = [torch.zeros(1, 1, 1, 16)] * 3
    if function == "flash_attn_varlen_func":
        args += [torch.tensor([0, 1], dtype=torch.int32)] * 2
    with patch.object(torch.version, "hip", "test"):
        with pytest.raises(NotImplementedError, match="mask_mod"):
            getattr(module, function)(*args, ver=4, mask_mod=object())


def test_fused_projection_ratio_keeps_cuda_and_cpu_routes():
    import sglang

    source = Path(sglang.__file__).parent / "srt/models/qwen3_5.py"
    tree = ast.parse(source.read_text())
    value = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_GDN_FUSED_QKVZBA_RATIOS"
            for t in n.targets
        )
    )
    code = compile(ast.Expression(value), str(source), "eval")
    assert eval(code, {"_use_aiter": False, "_is_cuda": True}) == (1, 2, 3, 4)
    assert eval(code, {"_use_aiter": False, "_is_cuda": False}) == (1, 2, 4)
    assert eval(code, {"_use_aiter": True, "_is_cuda": False}) == (1, 2, 3, 4, 8)

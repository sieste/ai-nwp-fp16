import copy

import torch

from nwpprec.precision import cast, cast_inputs, convert, keep_fp32


def net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 4))


def test_cast_leaves_non_float_alone():
    out = cast({"a": torch.ones(2), "b": (torch.arange(3), 1.5, None)}, torch.bfloat16)
    assert out["a"].dtype == torch.bfloat16 and out["b"][0].dtype == torch.int64 and out["b"][1:] == (1.5, None)


def test_convert_boundaries():
    m = net()
    convert(m[1], torch.bfloat16)
    seen = {}
    m[1].register_forward_hook(lambda mod, a, o: seen.update(inp=a[0].dtype, out=o.dtype))
    y = m(torch.randn(3, 4))
    assert m[0].weight.dtype == torch.float32 and m[1].weight.dtype == torch.bfloat16
    assert seen["out"] == torch.float32 and y.dtype == torch.float32  # hook sees the wrapped output


def test_convert_whole_equals_manual_bf16():
    m, ref = net(), net().bfloat16()
    convert(m, torch.bfloat16)
    x = torch.randn(3, 4)
    assert torch.equal(m(x), ref(x.bfloat16()).float())


def test_cast_inputs_only():
    m = net().bfloat16()
    cast_inputs(m, torch.bfloat16)
    assert m(torch.randn(3, 4)).dtype == torch.bfloat16


def test_keep_fp32_restores_original_weights():
    m = net()
    fp32 = copy.deepcopy(m[1].state_dict())
    convert(m, torch.bfloat16)
    keep_fp32(m[1], fp32, torch.bfloat16)
    assert torch.equal(m[1].weight, fp32["weight"])  # not the bf16-rounded values
    assert m(torch.randn(3, 4)).dtype == torch.float32

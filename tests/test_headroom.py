import torch

from nwpprec.headroom import Headroom


class Fixed(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([1.0, -500.0, float("inf"), 1e-6, 0.0, 2.0], dtype=torch.float16) * x


def test_headroom_statistics():
    model = torch.nn.Sequential(Fixed())
    h = Headroom(model, n_steps=2)
    with torch.inference_mode():
        model(torch.tensor(1.0, dtype=torch.float16))
        h.step = 1
        model(torch.tensor(0.5, dtype=torch.float16))
    s = h.summary()
    assert s["max_abs"] == 500 and s["step_max_abs"] == [500, 250]
    assert s["overflowing_modules"] == ["0", "<model>"]
    assert s["modules"]["0"]["subnormal_frac"] == 2 / 12  # 1e-6 and 0.5e-6

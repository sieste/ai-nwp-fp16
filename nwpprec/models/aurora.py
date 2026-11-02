"""Aurora 1.5 (0.25 degrees, 6-hour steps), deterministic or ensemble
checkpoint, via the microsoft-aurora package.

Aurora normalises the batch inside its forward pass and casts the batch to
the dtype of its first parameter before doing so. In bf16 and fp16, an fp32
parameter registered first keeps the batch and its normalisation in fp32,
and the encoder and decoder round their (normalised) inputs on entry; Aurora
casts its prediction to fp32 before denormalising it.

In fp32 the model does not fit in 16 GB: the backbone's transformer blocks
are kept on the CPU and moved to the GPU one at a time. In every precision,
the decoder's level aggregation runs in chunks of grid positions, which
lowers the peak memory without changing the result (positions are
independent there).
"""

import dataclasses
import pickle
from datetime import timedelta

import numpy as np
import torch
from aurora import AuroraV1p5, AuroraV1p5Ensemble, Batch, Metadata, rollout
from aurora.insolation import insolation

from .. import era5
from ..paths import CHECKPOINTS
from ..precision import CASTERS, DTYPES, cast_inputs, convert

STEP_HOURS = 6
LAT, LON = era5.LAT[:720], era5.LON  # Aurora's output leaves out the south pole row
SKIP_NAN = False
VARIANTS = {"deterministic": (AuroraV1p5, "aurora-0.25-v1.5.ckpt"),
            "ensemble": (AuroraV1p5Ensemble, "aurora-0.25-v1.5-ensemble.ckpt")}
STATIC = CHECKPOINTS / "aurora-0.25-v1.5-static.pickle"
SURFACE = {"t2m": "2t", "u10": "10u", "v10": "10v", "msl": "msl", "d2m": "2d", "tcwv": "tcwv", "tcc": "tcc",
           "u100": "100u", "v100": "100v", "sp": "sp", "lcc": "lcc", "mcc": "mcc", "hcc": "hcc", "skt": "skt",
           "stl1": "stl1", "swvl1": "swvl1", "siconc": "ci", "sd": "scaled_sd"}  # ERA5 name -> Aurora name
ATMOS = ("t", "u", "v", "q", "z")
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
FIELDS = {"2t": ("surf", "2t", None), "msl": ("surf", "msl", None), "10u": ("surf", "10u", None),
          "10v": ("surf", "10v", None), "z500": ("atmos", "z", 500), "t850": ("atmos", "t", 850)}
CHUNK = 4096  # grid positions per chunk in the decoder
COMPONENTS = ("encoder",) + tuple(f"backbone_stage_{i}" for i in range(6)) + ("decoder",)


def _batch_type(self, dtype):
    """Batch.type without casting the latitudes and longitudes."""
    return dataclasses.replace(self, surf_vars={k: v.type(dtype) for k, v in self.surf_vars.items()},
                               static_vars={k: v.type(dtype) for k, v in self.static_vars.items()},
                               atmos_vars={k: v.type(dtype) for k, v in self.atmos_vars.items()})


Batch.type = _batch_type
CASTERS[Batch] = _batch_type


def _chunk_decoder(model) -> None:
    attn, ff, _, _ = model.decoder.level_decoder.layers[0]
    ff_forward, attn_forward = ff.forward, attn.forward

    def ff_chunked(x):
        if x.shape[0] <= CHUNK:
            return ff_forward(x)
        return torch.cat([ff_forward(x[i:i + CHUNK]) for i in range(0, x.shape[0], CHUNK)], dim=0)

    def attn_chunked(latents, x):
        if latents.shape[0] <= CHUNK:
            return attn_forward(latents, x)
        return torch.cat([attn_forward(latents[i:i + CHUNK], x[i:i + CHUNK]) for i in range(0, latents.shape[0], CHUNK)], dim=0)

    ff.forward, attn.forward = ff_chunked, attn_chunked


def _stream_backbone(model, device) -> None:
    def to_gpu(block, args):
        block.to(device)

    def to_cpu(block, args, output):
        block.to("cpu")
        torch.cuda.empty_cache()

    model.encoder.to(device)
    model.decoder.to(device)
    model.backbone.to(device)
    for stage in list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers):
        stage.blocks.to("cpu")
        for block in stage.blocks:
            block.register_forward_pre_hook(to_gpu)
            block.register_forward_hook(to_cpu)


def components(model) -> dict:
    stages = list(model.backbone.encoder_layers) + list(model.backbone.decoder_layers)
    return {"encoder": model.encoder, **{f"backbone_stage_{i}": s for i, s in enumerate(stages)}, "decoder": model.decoder}


def _keep_state_fp32(model, device) -> None:
    """An fp32 parameter registered first: Aurora casts the batch to the dtype of its first parameter."""
    model.register_parameter("_fp32_state", torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False))


def load(precision: str, device: str = "cuda", variant: str = "deterministic", components_: tuple | None = None):
    """components_: convert only these components to `precision`, the rest of
    the model staying fp32 (the one-at-a-time experiments)."""
    cls, checkpoint = VARIANTS[variant]
    model = cls(autocast=False, use_fp16_safe_attention=False)
    model.load_checkpoint_local(str(CHECKPOINTS / checkpoint))
    model.eval()
    if precision == "fp32" or components_ is not None:
        _stream_backbone(model, device)
        _chunk_decoder(model)
        if components_ is not None:
            for name, module in components(model).items():
                if name in components_:
                    convert(module, DTYPES[precision])
            _keep_state_fp32(model, device)
        return model
    dtype = DTYPES[precision]
    model = model.to(dtype).to(device)
    _chunk_decoder(model)
    _keep_state_fp32(model, device)
    cast_inputs(model.encoder, dtype)
    cast_inputs(model.decoder, dtype)
    return model


def initial_state(init_time) -> Batch:
    times = [init_time - timedelta(hours=STEP_HOURS), init_time]
    x = [era5.inputs(t, surface=tuple(SURFACE), atmos=ATMOS, levels=LEVELS) for t in times]
    surf = {SURFACE[k]: torch.from_numpy(np.stack([xt[k] for xt in x])[None]) for k in SURFACE}
    lat, lon = era5.LAT.astype(np.float32), era5.LON.astype(np.float32)
    surf["insolation"] = torch.from_numpy(
        np.stack([insolation([t], lat, lon, enforce_2d=True)[0] for t in times])[None].astype(np.float32))
    atmos = {k: torch.from_numpy(np.stack([xt[k] for xt in x])[None]) for k in ATMOS}
    with open(STATIC, "rb") as f:
        static = {k: torch.from_numpy(v) for k, v in pickle.load(f).items()}
    return Batch(surf_vars=surf, static_vars=static, atmos_vars=atmos,
                 metadata=Metadata(lat=torch.from_numpy(era5.LAT.copy()), lon=torch.from_numpy(era5.LON.copy()),
                                   time=(init_time,), atmos_levels=LEVELS))


def forecast(model, init_time, max_lead: int = 240, seed: int = 0):
    """Yields (lead hours, prediction batch) every 6 hours. `seed` sets the
    ensemble checkpoint's noise."""
    batch = initial_state(init_time)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.reset_noise()
    with torch.inference_mode():
        for step, pred in enumerate(rollout(model, batch, steps=max_lead // STEP_HOURS)):
            yield STEP_HOURS * (step + 1), pred


def fields(pred) -> dict:
    """{verification variable: (720, 1440) float32 array}."""
    out = {}
    for var, (kind, name, level) in FIELDS.items():
        x = pred.surf_vars[name][0, 0] if kind == "surf" else pred.atmos_vars[name][0, 0, LEVELS.index(level)]
        out[var] = x.float().cpu().numpy()
    return out

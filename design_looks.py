"""Build the D Classic / CPM35 approximations from anchors measured in the sample photos (no iPhone needed).

Each anchor = (assumed camera input RGB, measured filtered RGB) for an object in the reference photos
(private photos of people, not included in the repo).
Inputs are guesses for what a phone would have recorded (whites under warm venue light etc.), so the overall
warmth is uncertain -> `cast` (0..1) blends from a neutral version of the curves to the full measured cast.

usage: python design_looks.py [--cast 1.0]
"""
import argparse
import json

import numpy as np
from scipy.interpolate import PchipInterpolator

from common import LUT_SIZE, ROOT, apply_lut, delta_e, lum, save_cube

LOOKS = {
    # photo 3 (night, direct flash): soft, faded, warm; matches the public "CPM35 = soft fades, slight blur" notes
    "cpm35": {
        "anchors": [
            ((0.00, 0.00, 0.00), (0.247, 0.165, 0.030), "darkest pixels (p0.5)"),
            ((0.08, 0.075, 0.07), (0.298, 0.216, 0.119), "black crop top, flash-lit"),
            ((0.45, 0.10, 0.12), (0.482, 0.204, 0.128), "maroon skirt"),
            ((0.78, 0.55, 0.42), (0.842, 0.527, 0.213), "face skin"),
            ((0.75, 0.72, 0.66), (0.820, 0.703, 0.495), "stone floor, flash-lit"),
            ((0.88, 0.85, 0.80), (0.916, 0.767, 0.565), "white skirt"),
            ((0.95, 0.93, 0.88), (0.965, 0.817, 0.660), "white sleeve"),
            ((1.00, 1.00, 1.00), (1.000, 0.920, 0.800), "brightest (p99.5)"),
        ],
        "saturation": 0.9,
        "blur_sigma": 1.3 / 960,
        "bloom": {"sigma": 0.012, "threshold": 0.8, "strength": [0.22, 0.17, 0.10]},
        "vignette": {"a": [-0.12, -0.14, -0.18], "b": [-0.05, -0.05, -0.06]},
        "grain": {"sigma": 1.0 / 3024, "mono": 0.85, "levels": [0.1, 0.5, 0.9],
                  "std": [[0.016, 0.016, 0.018], [0.013, 0.013, 0.014], [0.007, 0.007, 0.008]]},
    },
    # photos 1-2: crisp, saturated, golden; blue crushed hard, red-lifted blacks
    "dclassic": {
        "anchors": [
            ((0.00, 0.00, 0.00), (0.215, 0.070, 0.000), "darkest pixels (p0.5)"),
            ((0.08, 0.07, 0.06), (0.294, 0.162, 0.004), "dark shoe"),
            ((0.55, 0.42, 0.30), (0.574, 0.352, 0.051), "stone wall, warm light"),
            ((0.80, 0.58, 0.45), (0.838, 0.535, 0.099), "arm skin"),
            ((0.97, 0.90, 0.75), (0.986, 0.659, 0.174), "white kurta, sunlit"),
            ((1.00, 1.00, 1.00), (1.000, 0.800, 0.400), "brightest (p99.5)"),
        ],
        "saturation": 1.1,
        "blur_sigma": 0.0,
        "bloom": {"sigma": 0.0, "threshold": 1.0, "strength": [0.0, 0.0, 0.0]},
        "vignette": {"a": [-0.10, -0.11, -0.13], "b": [-0.04, -0.04, -0.05]},
        "grain": {"sigma": 0.8 / 3024, "mono": 0.9, "levels": [0.1, 0.5, 0.9],
                  "std": [[0.009, 0.009, 0.010], [0.007, 0.007, 0.008], [0.004, 0.004, 0.004]]},
    },
}


def channel_curves(anchors):
    curves = []
    for c in range(3):
        pts = sorted({a[0][c]: a[1][c] for a in anchors}.items())
        x, y = np.array(pts).T
        y = np.maximum.accumulate(y)  # keep monotone
        curves.append(PchipInterpolator(x, y))
    return curves


def build_lut(look, cast):
    curves = channel_curves(look["anchors"])
    v = np.linspace(0, 1, LUT_SIZE)
    r, g, b = np.meshgrid(v, v, v, indexing="ij")
    rgb = np.stack([r, g, b], -1)
    out = np.stack([curves[c](rgb[..., c]) for c in range(3)], -1)
    if cast < 1:
        # neutral version: same per-pixel luminance through the mean curve, keeping the input hue
        y_out = lum(out)[..., None]
        y_in = lum(rgb)[..., None]
        neutral = np.clip(rgb * (y_out / np.maximum(y_in, 1e-4)), 0, 1)
        neutral[np.squeeze(y_in, -1) < 1e-4] = y_out[np.squeeze(y_in, -1) < 1e-4]
        out = cast * out + (1 - cast) * neutral
    y = lum(out)[..., None]
    out = y + look["saturation"] * (out - y)
    return np.clip(out, 0, 1).astype(np.float32)


def main(cast, suffix=""):
    (ROOT / "presets").mkdir(exist_ok=True)
    for name, look in LOOKS.items():
        lut = build_lut(look, cast)
        full = name + suffix
        save_cube(ROOT / "presets" / f"{full}.cube", lut, full)
        preset = {"name": full, "lut": f"{full}.cube", "cast": cast, "source": "estimated from sample photos",
                  **{k: look[k] for k in ("blur_sigma", "bloom", "vignette", "grain")}}
        (ROOT / "presets" / f"{full}.json").write_text(json.dumps(preset, indent=2))
        if cast == 1:
            ins = np.array([a[0] for a in look["anchors"]], np.float32)[None]
            outs = np.array([a[1] for a in look["anchors"]], np.float32)[None]
            de = delta_e(apply_lut(ins, lut), outs)[0]
            worst = look["anchors"][int(de.argmax())][2]
            print(f"{full}: anchor reproduction mean dE2000 {de.mean():.2f}, worst {de.max():.2f} ({worst})")
        print("wrote presets/" + full + ".cube/.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cast", type=float, default=1.0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    main(a.cast, a.suffix)

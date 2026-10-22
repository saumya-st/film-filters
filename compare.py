"""Compare our preset against a real Dazz export of the same photo.

usage: python compare.py <original photo> <dazz export of it> <preset name>
Writes report/<original>_<preset>_compare.jpg (original | ours | dazz | dE heatmap) and prints dE2000 stats.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from apply_filter import load_preset
from common import ROOT, apply_preset, delta_e, read_rgb, write_rgb


def small(img, w=600):
    h = round(img.shape[0] * w / img.shape[1])
    return cv2.GaussianBlur(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 1.0)


def main(orig_path, dazz_path, name):
    orig, dazz = read_rgb(orig_path), read_rgb(dazz_path)
    if abs(orig.shape[1] / orig.shape[0] - dazz.shape[1] / dazz.shape[0]) > 0.02:
        print(f"warning: aspect differs ({orig.shape[1]}x{orig.shape[0]} vs {dazz.shape[1]}x{dazz.shape[0]})")
    preset, lut = load_preset(name)
    ours = apply_preset(orig, preset, lut, grain=False)
    de = delta_e(small(ours), small(dazz))
    print(f"dE2000 (grain removed): mean={de.mean():.2f} median={np.median(de):.2f} 95%={np.percentile(de, 95):.2f}")

    ours_g = apply_preset(orig, preset, lut, grain=True)
    th = 1000
    tiles = [cv2.resize(i, (round(i.shape[1] * th / i.shape[0]), th), interpolation=cv2.INTER_AREA)
             for i in (orig, ours_g, dazz)]
    heat = cv2.applyColorMap((np.clip(de / 10, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    tiles.append(cv2.resize(heat.astype(np.float32) / 255, tiles[0].shape[1::-1], interpolation=cv2.INTER_NEAREST))
    out = ROOT / "report" / f"{Path(orig_path).stem}_{name}_compare.jpg"
    write_rgb(out, np.concatenate(tiles, 1), quality=90)
    print("wrote", out, "(original | ours | dazz | dE heatmap, 0..10)")
    return de


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])

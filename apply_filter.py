"""Apply a measured preset to a photo.

usage: python apply_filter.py <input> <preset name> <output> [--no-grain] [--date "'26 10 22"] [--seed N]
"""
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, apply_preset, load_cube, read_rgb, write_rgb


def load_preset(name):
    p = json.loads((ROOT / "presets" / f"{name}.json").read_text())
    return p, load_cube(ROOT / "presets" / p["lut"])


def date_stamp(img, text):
    h, w = img.shape[:2]
    size = int(h * 0.035)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", size)
    except OSError:
        font = ImageFont.load_default(size)
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    tw, th = d.textbbox((0, 0), text, font=font)[2:]
    d.text((w - tw - int(w * 0.06), h - th - int(h * 0.05)), text, fill=255, font=font)
    m = np.asarray(layer, np.float32) / 255
    glow = cv2.GaussianBlur(m, (0, 0), size * 0.15)
    color = np.array([1.0, 0.55, 0.15], np.float32)
    a = np.clip(m * 0.9 + glow * 0.6, 0, 1)[..., None]
    return img * (1 - a) + np.maximum(img, color) * a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input"), ap.add_argument("preset"), ap.add_argument("output")
    ap.add_argument("--no-grain", action="store_true")
    ap.add_argument("--date")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    preset, lut = load_preset(a.preset)
    out = apply_preset(read_rgb(a.input), preset, lut, grain=not a.no_grain, seed=a.seed)
    if a.date:
        out = date_stamp(out, a.date)
    write_rgb(a.output, out)
    print("wrote", a.output)

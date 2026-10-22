"""Render the README figures into docs/ (test charts, looks on Kodak test photos, self-test comparison)."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from apply_filter import load_preset
from common import ROOT, apply_preset, make_color_chart, make_edge_chart, read_rgb, write_rgb
from synth_test import kodak_image

DOCS = ROOT / "docs"


def label(img, text):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 22)
    except OSError:
        font = ImageFont.load_default(22)
    d.rectangle((0, 0, d.textlength(text, font=font) + 20, 34), fill=(0, 0, 0))
    d.text((10, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(im, np.float32) / 255


def pad(img, p=4):
    return np.pad(img, ((p, p), (p, p), (0, 0)), constant_values=1)


def main():
    DOCS.mkdir(exist_ok=True)
    charts = [cv2.resize(c, (378, 504), interpolation=cv2.INTER_AREA) for c in (make_color_chart("A"), make_edge_chart())]
    write_rgb(DOCS / "charts.jpg", np.concatenate([pad(c) for c in charts], 1), quality=90)

    looks = [("Original", None), ("Soft 35", "cpm35_soft"), ("Golden", "dclassic_soft")]
    presets = {p: load_preset(p) for _, p in looks if p}
    rows = []
    for name in ("kodim05", "kodim15", "kodim23"):
        im = cv2.resize(read_rgb(kodak_image(name)), (480, 320), interpolation=cv2.INTER_AREA)
        tiles = [im if p is None else apply_preset(im, *presets[p], seed=1) for _, p in looks]
        if not rows:
            tiles = [label(t, n) for t, (n, _) in zip(tiles, looks)]
        rows.append(np.concatenate([pad(t) for t in tiles], 1))
    write_rgb(DOCS / "looks.jpg", np.concatenate(rows, 0), quality=90)

    compare = ROOT / "report" / "kodim04_synth_compare.jpg"
    if compare.exists():
        img = read_rgb(compare)
        w = img.shape[1] // 4
        names = ["Original", "Recovered by analyze.py", "Known look (ground truth)", "dE2000 heatmap (0-10)"]
        tiles = [label(img[:, i * w:(i + 1) * w], n) for i, n in enumerate(names)]
        write_rgb(DOCS / "selftest.jpg", np.concatenate(tiles, 1), quality=90)
    print("wrote", sorted(p.name for p in DOCS.iterdir()))


if __name__ == "__main__":
    main()

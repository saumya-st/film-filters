"""Self-test: fake a 'Dazz export' with a known film look, then check analyze.py recovers it."""
import json

import cv2
import numpy as np

import analyze
import compare
from common import LUT_SIZE, ROOT, apply_lut, apply_preset, delta_e, load_cube, lum, save_cube, write_rgb
from make_charts import all_charts

SYN = ROOT / "synth"


def truth_color(c):
    y = lum(c)[..., None]
    c = y + 0.8 * (c - y)                                    # desaturate
    c = 0.06 + 0.9 * (0.5 * c + 0.5 * c * c * (3 - 2 * c))   # lifted blacks + S-curve
    c = c * np.array([1.06, 1.0, 0.82])                      # warm
    c[..., 1] += 0.04 * (1 - y[..., 0]) ** 2                 # green-ish shadows
    return np.clip(c, 0, 1)


TRUTH = {
    "name": "truth", "lut": "truth.cube", "blur_sigma": 1.4 / 3024,
    "bloom": {"sigma": 35 / 3024, "threshold": 0.75, "strength": [0.35, 0.22, 0.12]},
    "vignette": {"a": [-0.20, -0.22, -0.26], "b": [-0.10, -0.10, -0.12]},
    "grain": {"sigma": 1.0 / 3024, "mono": 0.7, "levels": [0.1, 0.5, 0.9], "std": [[.03, .03, .035], [.025, .025, .028], [.012, .012, .014]]},
}


def fake_export(img, rotate):
    h, w = img.shape[:2]
    if rotate:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 1.2, 1.02)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(img, (round(w * 0.8), round(h * 0.8)), interpolation=cv2.INTER_AREA)


def main():
    (ROOT / "presets").mkdir(exist_ok=True)
    v = np.linspace(0, 1, LUT_SIZE)
    r, g, b = np.meshgrid(v, v, v, indexing="ij")
    truth_lut = truth_color(np.stack([r, g, b], -1)).astype(np.float32)
    save_cube(ROOT / "presets" / "truth.cube", truth_lut, "truth")
    (ROOT / "presets" / "truth.json").write_text(json.dumps(TRUTH, indent=2))

    exp = SYN / "exports"
    for i, (name, img) in enumerate(sorted(all_charts().items(), key=lambda t: hash(t[0]) % 7)):
        out = apply_preset(img, TRUTH, truth_lut, seed=i)
        write_rgb(exp / f"IMG_{4100 + i}.jpg", fake_export(out, rotate=name.startswith("chart")), quality=90)
    print("fake exports written")

    analyze.main(exp, "synth")
    rec = json.loads((ROOT / "presets" / "synth.json").read_text())
    rec_lut = load_cube(ROOT / "presets" / "synth.cube")

    rnd = np.random.default_rng(3).random((1, 4000, 3)).astype(np.float32)
    de = delta_e(apply_lut(rnd, rec_lut), apply_lut(rnd, truth_lut))
    print(f"\nLUT recovery on 4000 random colours: mean dE2000={de.mean():.2f}, 95%={np.percentile(de, 95):.2f}")
    for k in ("blur_sigma",):
        print(f"{k}: truth {TRUTH[k] * 3024:.2f}px  recovered {rec[k] * 3024:.2f}px")
    tb, rb = TRUTH["bloom"], rec["bloom"]
    print(f"bloom sigma: truth {tb['sigma'] * 3024:.1f}px recovered {rb['sigma'] * 3024:.1f}px; "
          f"thr {tb['threshold']} vs {rb['threshold']:.2f}; strength {tb['strength']} vs {np.round(rb['strength'], 3).tolist()}")
    print(f"vignette a: {TRUTH['vignette']['a']} vs {np.round(rec['vignette']['a'], 3).tolist()}  "
          f"b: {TRUTH['vignette']['b']} vs {np.round(rec['vignette']['b'], 3).tolist()}")
    print(f"grain sigma: 1.00px vs {rec['grain']['sigma'] * 3024:.2f}px, mono 0.7 vs {rec['grain']['mono']:.2f}")

    # end-to-end on a real photo: "dazz" = truth applied, compare with recovered preset
    src = next((ROOT.parent / "Desktop" / "filter").glob("*00.05.24*"), None)
    if src is not None:
        from common import read_rgb
        photo = read_rgb(src)
        write_rgb(SYN / "photo_truth.jpg", apply_preset(photo, TRUTH, truth_lut, seed=9), quality=92)
        compare.main(src, SYN / "photo_truth.jpg", "synth")


if __name__ == "__main__":
    main()

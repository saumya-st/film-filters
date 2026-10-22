"""Measure a filter from Dazz exports of the test charts -> presets/<name>.cube + presets/<name>.json.

usage: python analyze.py <folder with the exported charts> <preset name>
File names don't matter: charts are recognised by matching against the known test images, flats by brightness.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import curve_fit
from scipy.special import erf

from common import (ARUCO, CHART_IDS, COLS, DOT_COLS, DOT_LEVELS, DOT_R, DOT_ROWS, EDGE_HI, EDGE_LO, EDGE_X, H,
                    IMAGE_EXTS, LUT_SIZE, PITCH, ROOT, ROWS, W, apply_lut, big_blur, chart_perm, delta_e, grain_noise,
                    lum, make_color_chart, make_edge_chart, marker_corners, patch_colors, radius_map, read_rgb, save_cube, slot_center)

GRAIN_HP = 3.0          # high-pass sigma (chart px) used to isolate grain
LEVEL_EDGES = np.array([0, .15, .3, .45, .6, .75, .9, 1.0])


def log(msg, lines):
    print(msg)
    lines.append(msg)


# ---------- loading & registration ----------
TEMPLATES = {"A": lambda: make_color_chart("A"), "B": lambda: make_color_chart("B"), "edge": make_edge_chart}


def marker_guess(gray, base):
    """Chart->export homography from ArUco markers if >=3 are found (optional; ECC does the real work)."""
    det = cv2.aruco.ArucoDetector(ARUCO, cv2.aruco.DetectorParameters())
    s = 0.25
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    small = cv2.threshold(cv2.GaussianBlur(small, (0, 0), 2), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    corners, ids, _ = det.detectMarkers(small)
    if ids is None:
        return None
    src, dst = [], []
    for c, i in zip(corners, ids.ravel()):
        if base <= i < base + 4:
            src.append(c.reshape(4, 2) / s)
            dst.append(marker_corners(i - base))
    if len(src) < 3:
        return None
    hm, _ = cv2.findHomography(np.concatenate(dst), np.concatenate(src), cv2.RANSAC, 8.0)
    return hm


def ecc(template, img_l, t, f, iters):
    """Refine chart->export homography t at pyramid scale f. Returns (correlation, t)."""
    tpl = cv2.resize(template, (round(W * f), round(H * f)), interpolation=cv2.INTER_AREA)
    inp = cv2.resize(img_l, (round(img_l.shape[1] * f), round(img_l.shape[0] * f)), interpolation=cv2.INTER_AREA)
    S = np.diag([f, f, 1.0])
    ts = (S @ t @ np.linalg.inv(S)).astype(np.float32)
    try:
        cc, ts = cv2.findTransformECC(tpl, inp, ts, cv2.MOTION_HOMOGRAPHY,
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6), None, 3)
    except cv2.error:
        return -1.0, t
    return cc, np.linalg.inv(S) @ ts.astype(np.float64) @ S


def load(folder):
    tpls = {k: lum(make()).astype(np.float32) for k, make in TEMPLATES.items()}
    charts, flats = {}, []
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        img = read_rgb(f)
        img_l = lum(img).astype(np.float32)
        h, w = img_l.shape
        scale = np.diag([w / W, h / H, 1.0])
        best = (0.6, None, None)  # minimum correlation to count as a chart
        for kind, tpl in tpls.items():
            t0 = marker_guess((img_l * 255).astype(np.uint8), CHART_IDS[kind])
            cc, t = ecc(tpl, img_l, scale if t0 is None else t0, 1 / 8, 60)
            if cc > best[0]:
                best = (cc, kind, t)
        cc, kind, t = best
        if kind is None:
            flats.append((float(img.mean()), f.name, img))
            continue
        for fs in (1 / 4, 1 / 2):
            cc, t = ecc(tpls[kind], img_l, t, fs, 50)
        charts[kind] = cv2.warpPerspective(img, t, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                           borderMode=cv2.BORDER_REFLECT)
        print(f"  {f.name}: {kind} chart (alignment correlation {cc:.3f})")
    flats.sort(key=lambda t: t[0])
    named = {}
    if len(flats) == 3:
        named = dict(zip(["black", "grey", "white"], flats))
    elif flats:
        print(f"  warning: expected 3 flat images, found {len(flats)}; using brightness guesses")
        for m, n, im in flats:
            named["black" if m < .2 else "white" if m > .8 else "grey"] = (m, n, im)
    for k, (_, n, im) in named.items():
        print(f"  {n}: flat {k}")
        if abs(im.shape[1] / im.shape[0] - W / H) > 0.02:
            print(f"  warning: {n} is not 3:4 ({im.shape[1]}x{im.shape[0]})")
    flats = {k: cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC) for k, (_, n, im) in named.items()}
    return charts, flats


# ---------- individual measurements ----------
def fit_vignette(grey):
    small = cv2.resize(grey, (W // 16, H // 16), interpolation=cv2.INTER_AREA)
    r = radius_map(*small.shape[:2])
    center = small[r < 0.15].mean(0)
    A = np.stack([r.ravel() ** 2, r.ravel() ** 4], 1)
    a, b = [], []
    for c in range(3):
        (ka, kb), *_ = np.linalg.lstsq(A, (small[..., c] / center[c]).ravel() - 1, rcond=None)
        a.append(float(ka))
        b.append(float(kb))
    return {"a": a, "b": b}


def vig_at(vig, x, y):
    r2 = ((x - (W - 1) / 2) ** 2 + (y - (H - 1) / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    return 1 + np.array(vig["a"]) * r2 + np.array(vig["b"]) * r2 ** 2


def fit_softness(edge):
    y0, y1 = 900, 1500
    prof = lum(edge[y0:y1, EDGE_X - 150:EDGE_X + 150]).mean(0)
    x = np.arange(len(prof)) - 150.0
    f = lambda x, lo, hi, mu, s: lo + (hi - lo) * 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * abs(s) + 1e-6)))
    (lo, hi, mu, s), _ = curve_fit(f, x, prof, p0=[prof[:30].mean(), prof[-30:].mean(), 0, 1.5])
    return abs(s), (lo, hi)


def radial_profile(img, cx, cy, rmax=300, step=3):
    box = img[cy - rmax:cy + rmax, cx - rmax:cx + rmax]
    yy, xx = np.mgrid[-rmax:rmax, -rmax:rmax]
    rr = np.hypot(xx, yy).ravel()
    bins = (rr // step).astype(int)
    nb = rmax // step
    keep = bins < nb
    out = np.zeros((nb, 3))
    cnt = np.bincount(bins[keep], minlength=nb)
    for c in range(3):
        out[:, c] = np.bincount(bins[keep], box[..., c].ravel()[keep], minlength=nb) / np.maximum(cnt, 1)
    return (np.arange(nb) + 0.5) * step, out


def fit_bloom(edge, soft_sigma, vig, lines):
    """Fit profile = background + strength * radial(gaussian_sigma * disc) around the bright discs."""
    profs = {}
    for lvl, cy in zip(DOT_LEVELS, DOT_ROWS):
        ps = [(radial_profile(edge, cx, cy), vig_at(vig, cx, cy)) for cx in DOT_COLS]
        r = ps[0][0][0]
        profs[lvl] = np.mean([p / v for (_, p), v in ps], 0)
    sel = r > DOT_R + 3 * soft_sigma + 6
    disc = np.zeros((600, 600), np.float32)
    cv2.circle(disc, (300, 300), DOT_R, 1.0, -1)

    def fit(prof, model):
        A = np.stack([np.ones(sel.sum()), model[sel]], 1)
        coef, res, *_ = np.linalg.lstsq(A, prof[sel], rcond=None)
        return coef[1], float(((A @ coef - prof[sel]) ** 2).sum())

    best = None
    for sig in np.geomspace(4, 150, 40):
        model = lum(radial_profile(np.repeat(cv2.GaussianBlur(disc, (0, 0), sig)[..., None], 3, 2), 300, 300)[1])
        amp, err = fit(profs[1.0], model)
        if best is None or err < best[2]:
            best = (sig, model, err, amp)
    sig, model = best[0], best[1]
    flat_err = float(((profs[1.0][sel] - profs[1.0][sel].mean(0)) ** 2).sum())
    strength = np.array(best[3])
    if not np.any(strength > 0.01) or best[2] > 0.8 * flat_err:
        log("  bloom: none detected", lines)
        return {"sigma": 0.0, "threshold": 1.0, "strength": [0.0, 0.0, 0.0]}
    a1 = float(lum(strength[None])[0])
    ratios = [(lvl, float(lum(np.array(fit(profs[lvl], model)[0])[None])[0]) / a1) for lvl in DOT_LEVELS[1:]]
    ths = [(lvl - q) / (1 - q) for lvl, q in ratios if 0.1 < q < 0.95]
    thr = float(np.clip(np.mean(ths), 0, 0.97)) if ths else 0.9
    strength = strength.clip(0).tolist()
    log(f"  bloom: sigma={sig:.1f}px threshold={thr:.2f} strength={np.round(strength, 3).tolist()}", lines)
    return {"sigma": float(sig) / W, "threshold": thr, "strength": strength}


def grain_estimator(img):
    """(std per channel, lag-1 autocorrelation of luminance) of the high-passed image."""
    res = img - cv2.GaussianBlur(img, (0, 0), GRAIN_HP)
    y = lum(res)
    rho = 0.5 * ((y[:, 1:] * y[:, :-1]).mean() + (y[1:] * y[:-1]).mean()) / y.var()
    return res.reshape(-1, 3).std(0), rho, res


def fit_grain_shape(grey):
    crop = grey[H // 2 - 600:H // 2 + 600, W // 2 - 600:W // 2 + 600]
    _, rho_meas, res = grain_estimator(crop)
    rr = res.reshape(-1, 3)
    mono = float(np.clip(np.mean([np.corrcoef(rr[:, i], rr[:, j])[0, 1] for i, j in ((0, 1), (1, 2), (0, 2))]), 0, 1))
    rng = np.random.default_rng(0)
    sigmas = np.array([0.0, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0])
    rhos, gains = [], []
    for s in sigmas:
        n = grain_noise(800, 800, s, rng)
        std, rho, _ = grain_estimator(np.repeat(n[..., None], 3, 2))
        rhos.append(rho)
        gains.append(std[0])
    sig = float(np.interp(rho_meas, rhos, sigmas))
    gain = float(np.interp(sig, sigmas, gains))
    return sig, gain, mono


# ---------- colour ----------
def measure_patches(chart, kind, res, vig, bloom_pred):
    cols = np.round(patch_colors() * 255) / 255
    out = np.zeros_like(cols)
    stds = np.zeros_like(cols)
    for s, ci in enumerate(chart_perm(kind)):
        cx, cy = slot_center(s)
        sl = (slice(cy - 30, cy + 30), slice(cx - 30, cx + 30))
        out[ci] = np.median(chart[sl].reshape(-1, 3), 0) / vig_at(vig, cx, cy) - bloom_pred[cy, cx]
        stds[ci] = res[sl].reshape(-1, 3).std(0)
    return cols, out, stds


def main(folder, name):
    lines = []
    out_dir = ROOT / "presets"
    out_dir.mkdir(exist_ok=True)
    print(f"loading {folder}")
    charts, flats = load(folder)
    missing = [k for k in ("A", "B", "edge") if k not in charts] + [k for k in ("grey", "black") if k not in flats]
    if missing:
        sys.exit(f"missing exports: {missing}")

    vig = fit_vignette(flats["grey"])
    log(f"  vignette a={np.round(vig['a'], 3).tolist()} b={np.round(vig['b'], 3).tolist()}", lines)

    soft, _ = fit_softness(charts["edge"])
    log(f"  softness sigma={soft:.2f}px (of {W})", lines)
    bloom = fit_bloom(charts["edge"], soft, vig, lines)

    blk = cv2.resize(flats["black"], (W // 32, H // 32), interpolation=cv2.INTER_AREA)
    leak = np.abs(blk - np.median(blk.reshape(-1, 3), 0)).max()
    log(f"  black level {np.round(np.median(blk.reshape(-1, 3), 0), 3).tolist()}, "
        f"max deviation on black (light leaks / vignette) {leak:.3f}", lines)

    gsig, ggain, mono = fit_grain_shape(flats["grey"])
    log(f"  grain sigma={gsig:.2f}px mono={mono:.2f}", lines)

    X, Y, S = [], [], []
    for kind in ("A", "B"):
        chart_in = make_color_chart(kind)
        if soft > 0.3:
            chart_in = cv2.GaussianBlur(chart_in, (0, 0), soft)
        h = np.clip((lum(chart_in) - bloom["threshold"]) / max(1e-3, 1 - bloom["threshold"]), 0, 1)
        bp = big_blur(h, bloom["sigma"] * W)[..., None] * np.array(bloom["strength"], np.float32)
        _, _, res = grain_estimator(charts[kind])
        x, y, s = measure_patches(charts[kind], kind, res, vig, bp)
        X.append(x), Y.append(y), S.append(s)
    de_ab = delta_e(Y[0][None], Y[1][None])
    log(f"  chart A vs B agreement: mean dE2000={de_ab.mean():.2f} (high = position-dependent effects)", lines)
    X, Y = X[0], (Y[0] + Y[1]) / 2

    rbf = RBFInterpolator(X, Y, kernel="thin_plate_spline", smoothing=1e-3)
    v = np.linspace(0, 1, LUT_SIZE)
    rg, gg, bg = np.meshgrid(v, v, v, indexing="ij")
    lut = np.clip(rbf(np.stack([rg.ravel(), gg.ravel(), bg.ravel()], 1)), 0, 1).reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)
    lut = lut.astype(np.float32)
    fit_de = delta_e(apply_lut(X[None], lut), Y[None])
    log(f"  LUT fit vs measured patches: mean dE2000={fit_de.mean():.2f}, 95%={np.percentile(fit_de, 95):.2f}", lines)

    # grain amplitude vs output luminance
    yl = lum(np.clip(Y, 0, 1))
    std = (S[0] + S[1]) / 2 / ggain
    centers, table = [], []
    for lo, hi in zip(LEVEL_EDGES[:-1], LEVEL_EDGES[1:]):
        m = (yl >= lo) & (yl < hi + 1e-6)
        if m.sum() >= 5:
            centers.append(float((lo + hi) / 2))
            table.append(np.median(std[m], 0).tolist())
    log(f"  grain std by level: " + ", ".join(f"{c:.2f}:{np.round(t, 3).tolist()}" for c, t in zip(centers, table)),
        lines)

    save_cube(out_dir / f"{name}.cube", lut, name)
    preset = {
        "name": name, "lut": f"{name}.cube",
        "blur_sigma": soft / W, "bloom": bloom, "vignette": vig,
        "grain": {"sigma": gsig / W, "mono": mono, "levels": centers, "std": table},
    }
    (out_dir / f"{name}.json").write_text(json.dumps(preset, indent=2))
    (out_dir / f"{name}_report.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / (name + '.json')} and {name}.cube")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

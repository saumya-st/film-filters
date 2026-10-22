"""Shared chart layout, image I/O and the film-look pipeline (mirrors the future GLSL shader)."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent

# ---- chart layout (all charts are 3024x4032, i.e. iPhone 3:4 so Dazz won't crop) ----
W, H = 3024, 4032
MARKER, QUIET = 300, 60
MARKER_TL = [(150, 150), (W - 450, 150), (W - 450, H - 450), (150, H - 450)]  # TL, TR, BR, BL
ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CHART_IDS = {"A": 0, "B": 4, "edge": 8}  # marker id base per chart; id = base + corner

PITCH, COLS, ROWS = 100, 27, 29
X0, Y0 = (W - COLS * PITCH) // 2, (H - ROWS * PITCH) // 2
LATTICE = 9

# edge/bloom test chart
EDGE_RECT = (300, 600, 2724, 1800)          # x0, y0, x1, y1 ; left half 0.3 grey, right half 0.6 grey
EDGE_X = 1512
EDGE_LO, EDGE_HI = 0.3, 0.6
DOT_R = 40
DOT_LEVELS = [1.0, 0.85, 0.7]
DOT_ROWS = [2150, 2650, 3150]
DOT_COLS = [600, 1200, 1800, 2400]

LUT_SIZE = 33


def patch_colors():
    """729 lattice colours + 32-step grey ramp + 22 random colours = 783 = COLS*ROWS."""
    v = np.arange(LATTICE) / (LATTICE - 1)
    b, g, r = np.meshgrid(v, v, v, indexing="ij")
    lattice = np.stack([r.ravel(), g.ravel(), b.ravel()], 1)
    grey = np.repeat((np.arange(32) / 31)[:, None], 3, 1)
    extra = np.random.default_rng(7).random((COLS * ROWS - len(lattice) - len(grey), 3))
    return np.concatenate([lattice, grey, extra]).astype(np.float32)


def chart_perm(kind):
    return np.random.default_rng({"A": 1, "B": 2}[kind]).permutation(COLS * ROWS)


def slot_center(s):
    return X0 + (s % COLS) * PITCH + PITCH // 2, Y0 + (s // COLS) * PITCH + PITCH // 2


def marker_corners(corner):
    x, y = MARKER_TL[corner]
    return np.array([[x, y], [x + MARKER, y], [x + MARKER, y + MARKER], [x, y + MARKER]], np.float32)


def draw_markers(img, base):
    for c, (x, y) in enumerate(MARKER_TL):
        img[y - QUIET:y + MARKER + QUIET, x - QUIET:x + MARKER + QUIET] = 1.0
        m = cv2.aruco.generateImageMarker(ARUCO, base + c, MARKER).astype(np.float32) / 255
        img[y:y + MARKER, x:x + MARKER] = m[..., None]


def make_color_chart(kind):
    img = np.full((H, W, 3), 0.5, np.float32)
    cols, perm = patch_colors(), chart_perm(kind)
    for s, ci in enumerate(perm):
        cx, cy = slot_center(s)
        img[cy - PITCH // 2:cy + PITCH // 2, cx - PITCH // 2:cx + PITCH // 2] = cols[ci]
    draw_markers(img, CHART_IDS[kind])
    return img


def make_edge_chart():
    img = np.zeros((H, W, 3), np.float32)
    x0, y0, x1, y1 = EDGE_RECT
    img[y0:y1, x0:EDGE_X] = EDGE_LO
    img[y0:y1, EDGE_X:x1] = EDGE_HI
    yy, xx = np.mgrid[:H, :W]
    for lvl, cy in zip(DOT_LEVELS, DOT_ROWS):
        for cx in DOT_COLS:
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= DOT_R ** 2] = lvl
    draw_markers(img, CHART_IDS["edge"])
    return img


# ---- I/O ----
def read_rgb(path):
    path = Path(path)
    if path.suffix.lower() in (".heic", ".heif"):
        import pillow_heif  # pip install pillow-heif
        pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(im, np.float32) / 255


def write_rgb(path, img, quality=95):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, quality=quality)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff"}


# ---- maths helpers ----
def lum(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def big_blur(img, sigma):
    """Gaussian blur for large sigmas: blur at reduced resolution then upsample."""
    if sigma <= 0:
        return img
    h, w = img.shape[:2]
    f = max(1.0, sigma / 6)
    small = cv2.resize(img, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def radius_map(h, w):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) / np.hypot(w / 2, h / 2)


def vignette_map(h, w, vig):
    r2 = radius_map(h, w) ** 2
    a, b = np.array(vig["a"], np.float32), np.array(vig["b"], np.float32)
    return 1 + r2[..., None] * a + (r2 ** 2)[..., None] * b


def grain_noise(h, w, sigma, rng):
    n = rng.standard_normal((h, w), dtype=np.float32)
    if sigma > 0.3:
        n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n / n.std()


# ---- LUT ----
def save_cube(path, lut, title):
    n = lut.shape[0]
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    f.write("{:.6f} {:.6f} {:.6f}\n".format(*lut[r, g, b]))


def load_cube(path):
    rows, n = [], None
    for line in open(path):
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        if t[0] == "LUT_3D_SIZE":
            n = int(t[1])
        elif t[0][0].isdigit() or t[0][0] in "-.":
            rows.append([float(x) for x in t[:3]])
    return np.array(rows, np.float32).reshape(n, n, n, 3).transpose(2, 1, 0, 3)  # -> [r,g,b,ch]


def apply_lut(img, lut, chunk_rows=512):
    """Trilinear lookup (what a GPU 3D texture does)."""
    n = lut.shape[0]
    out = np.empty_like(img)
    for y in range(0, img.shape[0], chunk_rows):
        part = np.clip(img[y:y + chunk_rows], 0, 1) * (n - 1)
        coords = part.reshape(-1, 3).T
        for c in range(3):
            out[y:y + chunk_rows, ..., c] = map_coordinates(lut[..., c], coords, order=1).reshape(part.shape[:2])
    return out


# ---- the film pipeline: blur -> LUT -> bloom -> vignette -> grain ----
def bloom_map(x, bloom, width):
    s = np.array(bloom["strength"], np.float32)
    if not np.any(s > 0):
        return np.zeros_like(x)
    t = bloom["threshold"]
    h = np.clip((lum(x) - t) / max(1e-3, 1 - t), 0, 1)
    return big_blur(h, bloom["sigma"] * width)[..., None] * s


def apply_preset(img, p, lut, grain=True, seed=0):
    h, w = img.shape[:2]
    x = img
    if p["blur_sigma"] * w > 0.3:
        x = cv2.GaussianBlur(x, (0, 0), p["blur_sigma"] * w)
    out = apply_lut(x, lut) + bloom_map(x, p["bloom"], w)
    out *= vignette_map(h, w, p["vignette"])
    if grain:
        g = p["grain"]
        rng = np.random.default_rng(seed)
        sig = g["sigma"] * w
        mono = grain_noise(h, w, sig, rng)
        y = lum(np.clip(out, 0, 1))
        km, kc = np.sqrt(g["mono"]), np.sqrt(1 - g["mono"])
        for c in range(3):
            amp = np.interp(y, g["levels"], np.array(g["std"])[:, c]).astype(np.float32)
            out[..., c] += amp * (km * mono + kc * grain_noise(h, w, sig, rng))
    return np.clip(out, 0, 1)


def delta_e(a, b):
    from skimage.color import deltaE_ciede2000, rgb2lab
    return deltaE_ciede2000(rgb2lab(np.clip(a, 0, 1)), rgb2lab(np.clip(b, 0, 1)))

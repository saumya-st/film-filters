"""Export presets as Android assets: luts/<id>_{soft,full}.bin (33^3 RGB8, r fastest) + looks.json.

usage: python export_android.py <android app/src/main/assets dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

from common import ROOT, load_cube

LOOKS = [("golden", "Golden", "dclassic"), ("soft35", "Soft 35", "cpm35")]


def main(assets):
    assets = Path(assets)
    (assets / "luts").mkdir(parents=True, exist_ok=True)
    out = []
    for look_id, label, preset in LOOKS:
        p = json.loads((ROOT / "presets" / f"{preset}.json").read_text())
        for variant, name in (("soft", f"{preset}_soft"), ("full", preset)):
            lut = load_cube(ROOT / "presets" / f"{name}.cube")            # [r,g,b,ch]
            data = np.round(lut.transpose(2, 1, 0, 3) * 255).astype(np.uint8)  # [b,g,r,ch] -> r fastest
            (assets / "luts" / f"{look_id}_{variant}.bin").write_bytes(data.tobytes())
        g = p["grain"]
        assert len(g["levels"]) == 3, "app shader expects 3 grain levels"
        out.append({
            "id": look_id, "label": label, "lutSize": int(lut.shape[0]),
            "blurSigma": p["blur_sigma"], "bloom": p["bloom"], "vignette": p["vignette"],
            "grain": {"mono": g["mono"], "levels": g["levels"], "std": g["std"]},
        })
    (assets / "looks.json").write_text(json.dumps(out, indent=2))
    print("exported", [o["id"] for o in out], "to", assets)


if __name__ == "__main__":
    main(sys.argv[1])

"""Generate the test images to import into Dazz Cam."""
import numpy as np

from common import H, ROOT, W, make_color_chart, make_edge_chart, write_rgb


def all_charts():
    return {
        "chart_color_A": make_color_chart("A"),
        "chart_color_B": make_color_chart("B"),
        "chart_edge": make_edge_chart(),
        "flat_grey": np.full((H, W, 3), 0.5, np.float32),
        "flat_black": np.zeros((H, W, 3), np.float32),
        "flat_white": np.ones((H, W, 3), np.float32),
    }


if __name__ == "__main__":
    out = ROOT / "charts"
    for name, img in all_charts().items():
        write_rgb(out / f"{name}.png", img)
        print("wrote", out / f"{name}.png")

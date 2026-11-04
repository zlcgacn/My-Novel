r"""把偏青蓝的裙子调回定稿的深绿。用 ComfyUI 自带的 Python 跑，不用开 ComfyUI。

用法：
    python_embeded\python.exe 裙子调色.py 输出目录 图1.png 图2.png ...

e5 出的近景裙子色相在 199° 左右（青蓝），全身 180°，定稿是 161°、亮度 0.29。拿偏色的图进素材，
LoRA 会把 dark green dress 学成青蓝。这里只动青蓝那一段色相、中等饱和、不太亮的像素（裙子），
色相整体挪到 161°，亮度按整块的中位数压到 0.30；天空太亮、头发饱和度太低，都不动。
输出 输出目录/<原名>.png（同名，方便直接替换进素材）。
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

TARGET_H, TARGET_V = 161 / 360, 0.30
H_LO, H_HI = 168 / 360, 218 / 360
S_MIN, V_MIN, V_MAX = 0.18, 0.12, 0.62


def recolor(im):
    hsv = np.asarray(im.convert("HSV")).astype(float) / 255
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (h >= H_LO) & (h <= H_HI) & (s >= S_MIN) & (v >= V_MIN) & (v <= V_MAX)
    if m.mean() < 0.03:
        return None
    # 边缘羽化一点，免得裙子和领子、背景接的地方起锯齿
    w = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))).astype(float) / 255
    scale = min(1.0, max(0.6, TARGET_V / np.median(v[m])))
    out = hsv.copy()
    out[..., 0] = h * (1 - w) + TARGET_H * w
    out[..., 2] = v * (1 - w) + v * scale * w
    return Image.fromarray((out * 255).round().astype(np.uint8), "HSV").convert("RGB")


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in map(Path, sys.argv[2:]):
        im = Image.open(src).convert("RGB")
        res = recolor(im)
        (res or im).save(out_dir / src.name)
        print(src.name, "调了" if res else "没找到裙子，原样", flush=True)


if __name__ == "__main__":
    main()

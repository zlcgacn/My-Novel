r"""给露耳朵的图在耳朵的位置重画头发。用 ComfyUI 自带的 Python 跑。

用法：
    python_embeded\python.exe 盖耳朵.py 输出目录 图1.png 图2.png ...
    python_embeded\python.exe 盖耳朵.py --wide 输出目录 ...   左右两框圈得更宽（侧脸）
    python_embeded\python.exe 盖耳朵.py --find 输出目录 ...   按颜色自动找耳朵（试验，见下）
    python_embeded\python.exe 盖耳朵.py --show 输出目录 ...   只画出 --find 找到的耳朵框，不写任务

默认：脸框左右两侧各圈一块（耳朵大致的位置）。正面、四分之三侧好用；结果是 <原名>_R_s9901_1.png。

--find（2026-11-03 试过，还不可靠）：人脸检测找到脸，在头部范围里找皮肤色的连通块；最大、压着脸框中心的那块是脸，
其余够大的、高度在眼睛到下巴之间的就是露出来的耳朵（被黑头发围住，所以和脸不连着）。
侧脸、正侧面的耳朵在头中间，也能找到。没找到耳朵的图原样跳过。
每只耳朵写一个 inpaint.py 的方框任务，一只接一只画；不挂 LoRA，只写头发，脸在框外不会动。
任务写到 输出目录/_盖耳朵.json；每张图最后一步的结果是 输出目录/<原名>_E<n>_s9901_1.png，
<原名>_done.txt 里写着它的最终文件名。正侧面的耳朵偏灰找不到，眼白、脖子偶尔会被当成耳朵，用之前先 --show 看一遍。
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

FACE_MODEL = r"D:\AI\ComfyUI_windows_portable\ComfyUI\models\ultralytics\bbox\face_yolov8m.pt"
PROMPT = ("black hair, long hair, straight hair, (hair over ears:1.5), hair strands, sidelocks, "
          "flat color, (no shading:1.2), matte")
NEG = "(ears:1.6), (ear:1.5), earring, hair behind ear, shiny hair, hair highlights, skin"


def find_ears(im, face):
    """返回耳朵框列表 [x0, y0, x1, y1]。"""
    x0, y0, x1, y1 = face
    w, h = x1 - x0, y1 - y0
    # 头部范围：脸框左右各放宽 0.7 个脸宽
    rx0, ry0 = int(max(0, x0 - 0.7 * w)), int(max(0, y0 - 0.2 * h))
    rx1, ry1 = int(min(im.width, x1 + 0.7 * w)), int(min(im.height, y1 + 0.05 * h))
    a = np.asarray(im.convert("RGB").crop((rx0, ry0, rx1, ry1))).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    # 皮肤：亮、偏粉（红比蓝高）；白领子 RGB 差不多，排除在外
    skin = (r > 170) & (r - b >= 10) & (r >= g)
    skin = ndimage.binary_opening(skin, iterations=1)
    lab, n = ndimage.label(skin)
    if n == 0:
        return []
    cx, cy = (x0 + x1) / 2 - rx0, (y0 + y1) / 2 - ry0
    face_lab = lab[int(cy), int(cx)]
    sizes = ndimage.sum(skin, lab, range(1, n + 1))
    if face_lab == 0:
        face_lab = int(np.argmax(sizes)) + 1
    dark = (a.max(axis=2) < 75)
    # 虹膜：深棕，红 > 绿 > 蓝
    iris = (r > 70) & (r < 190) & (r > g + 10) & (g > b) & (r - b > 30)
    ears = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        area = sizes[k - 1]
        if k == face_lab or not (0.006 * w * h <= area <= 0.08 * w * h):
            continue
        ys, xs = sl
        ey0, ey1, ex0, ex1 = ys.start + ry0, ys.stop + ry0, xs.start + rx0, xs.stop + rx0
        bw, bh = ex1 - ex0, ey1 - ey0
        mid_y = (ey0 + ey1) / 2
        # 耳朵在眼睛到下巴之间，竖长；太宽太扁的是脖子、脸颊碎块
        if not (y0 + 0.25 * h <= mid_y <= y1) or not (0.8 <= bh / max(bw, 1) <= 3.5):
            continue
        comp = lab == k
        # 四周大半是黑头发（眼白、脸颊碎块四周是皮肤，脖子挨着白领子）
        ring = ndimage.binary_dilation(comp, iterations=6) & ~ndimage.binary_dilation(comp, iterations=1)
        if dark[ring].mean() < 0.6:
            continue
        # 框里有虹膜的是眼睛
        if iris[ys, xs].mean() > 0.03:
            continue
        pad = 0.45 * max(ex1 - ex0, ey1 - ey0)
        ears.append([round(max(0, ex0 - pad)), round(max(0, ey0 - pad)),
                     round(min(im.width, ex1 + pad)), round(min(im.height, ey1 + pad))])
    return ears


def main():
    from ultralytics import YOLO

    args = sys.argv[1:]
    mode = args[0] if args[0].startswith("--") else ""
    if mode:
        args = args[1:]
    show = mode == "--show"
    reach = 0.5 if mode == "--wide" else 0.25
    out_dir = Path(args[0]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(FACE_MODEL)
    jobs, skipped = [], 0
    for src in map(Path, args[1:]):
        src = src.resolve()
        im = Image.open(src)
        boxes = model(str(src), verbose=False)[0].boxes
        if not len(boxes):
            print("没找到脸，跳过：", src.name)
            continue
        face = max(boxes.xyxy.tolist(), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        if mode in ("", "--wide"):
            fx0, fy0, fx1, fy1 = face
            fw, fh = fx1 - fx0, fy1 - fy0
            top, bottom = fy0 + 0.2 * fh, min(im.height, fy1 + 0.1 * fh)
            common = {"prompt": PROMPT, "negative_extra": NEG, "seed": 9901, "batch": 1, "denoise": 0.8}
            jobs.append(dict(common, name=f"{src.stem}_L", image=str(src),
                             box=[round(max(0, fx0 - reach * fw)), round(top), round(fx0 + 0.1 * fw), round(bottom)]))
            jobs.append(dict(common, name=f"{src.stem}_R", image=str(out_dir / f"{src.stem}_L_s9901_1.png"),
                             box=[round(fx1 - 0.1 * fw), round(top), round(min(im.width, fx1 + reach * fw)), round(bottom)]))
            continue
        ears = find_ears(im, face)
        if show:
            d = ImageDraw.Draw(im := im.convert("RGB"))
            d.rectangle(face, outline=(0, 160, 255), width=3)
            for e in ears:
                d.rectangle(e, outline=(255, 0, 0), width=4)
            im.save(out_dir / f"{src.stem}_耳朵框.png")
            continue
        if not ears:
            skipped += 1
            continue
        prev = str(src)
        for n, box in enumerate(ears, start=1):
            name = f"{src.stem}_E{n}"
            jobs.append({"name": name, "image": prev, "box": box, "prompt": PROMPT, "negative_extra": NEG,
                         "seed": 9901, "batch": 1, "denoise": 0.8})
            prev = str(out_dir / f"{name}_s9901_1.png")
        (out_dir / f"{src.stem}_done.txt").write_text(Path(prev).name, encoding="utf-8")
    if not show:
        task = out_dir / "_盖耳朵.json"
        task.write_text(json.dumps({"out_dir": ".", "jobs": jobs}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{len(jobs)} 只耳朵，{skipped} 张没露耳朵跳过，任务写到 {task}")


if __name__ == "__main__":
    main()

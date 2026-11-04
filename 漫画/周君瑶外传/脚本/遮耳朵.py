r"""把露出来的耳朵改成头发盖住。用 ComfyUI 自带的 Python 跑，ComfyUI 要开着。

用法：
    python_embeded\python.exe 遮耳朵.py [--lora 文件名] 输出目录 图1.png 图2.png ...

和 盖耳朵.py 的区别（那个留了方块接缝和粉色残迹，2026-11-03 弃用）：
1. 耳朵用 Grounding DINO 按文字找（见 找耳朵.py），只留脸旁边、眼睛到下巴高度的框，领口、脖子不算。
2. 从耳朵往下连成一条涂成头发色，原来的粉色就没了；再整张图重画，不裁不缩，没有方块边。
3. 蒙版羽化，配 Differential Diffusion：中间重画得多，往外越来越少，和原图自然接上。
4. 光靠提示词，模型看着下巴、脖子还会把耳朵画回来（试过，涂黑了也画回来）；
   所以拿原图的线稿，把那一条里的线擦掉、画上几条竖的发丝线，用 ControlNet（canny）压着，只能画成头发。
没找到耳朵的图原样跳过。结果是 输出目录/<原名>_遮.png，旁边的 <原名>_遮耳框.png 画着找到的框。
"""

import io
import sys
import time
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from comfy import CONTROLNET, NEGATIVE, get, upload  # noqa: E402
from inpaint import build, run  # noqa: E402
import 找耳朵  # noqa: E402

FACE_MODEL = r"D:\AI\ComfyUI_windows_portable\ComfyUI\models\ultralytics\bbox\face_yolov8m.pt"
PROMPT = ("zhoujunyao, 1girl, black hair, long hair, straight hair, (hair over ears:1.3), (sidelocks:1.2), hair strands, "
          "flat color, (no shading:1.2), matte")
NEG = "(ears:1.5), ear, earring, pink, skin, hair behind ear, shiny hair, hair highlights, purple hair, blue hair"


def ear_boxes(im, face):
    """Grounding DINO 找到的框里，留下在脸旁边的。"""
    fx0, fy0, fx1, fy1 = face
    fw, fh = fx1 - fx0, fy1 - fy0
    keep = []
    for (x0, y0, x1, y1), _ in 找耳朵.find_ears(im):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not (fy0 + 0.15 * fh <= cy <= fy1 + 0.05 * fh):
            continue
        if not (fx0 - 0.6 * fw <= cx <= fx1 + 0.6 * fw):
            continue
        if (x1 - x0) > 0.6 * fw or (y1 - y0) > 0.7 * fh:
            continue
        # 中心在脸框中间那块的、框里有深棕虹膜的，是眼睛
        if fx0 + 0.2 * fw <= cx <= fx1 - 0.2 * fw:
            continue
        a = np.asarray(im.crop((int(x0), int(y0), int(x1), int(y1)))).astype(int)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        if ((r > 70) & (r < 200) & (r > g + 10) & (g > b) & (r - b > 30)).mean() > 0.03:
            continue
        keep.append([x0, y0, x1, y1])
    return keep


def hair_color(im, hard):
    """涂色那一条外面一圈里偏暗的像素（头发）的中位色。取脸上方的头发会偏浅，补出来一块灰。"""
    m = np.asarray(hard) > 128
    k = np.ones((3, 3), np.uint8)
    ring = cv2.dilate(m.astype(np.uint8), k, iterations=15).astype(bool) & ~cv2.dilate(m.astype(np.uint8), k, iterations=3).astype(bool)
    a = np.asarray(im)[ring].reshape(-1, 3)
    lum = a.sum(axis=1)
    dark = a[lum <= np.percentile(lum, 40)]
    return tuple(int(v) for v in np.median(dark, axis=0))


def control_lines(im, hard, boxes, below=0.8, n=7):
    """原图的线稿，涂色那一条里的线全擦掉，换成一排竖的发丝线。
    只画 3 根时，耳朵框窄、旁边脸的轮廓线还在，模型会把空着的地方补成皮肤或耳朵；所以画密、从上面的头发接下来。"""
    edges = cv2.Canny(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY), 80, 160)
    m = cv2.dilate((np.asarray(hard) > 20).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
    edges[m > 0] = 0
    out = Image.fromarray(edges)
    d = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        for k in range(n):
            x = x0 - 0.1 * w + 1.2 * w * k / (n - 1)
            d.line([(x, y0 - 0.6 * h), (x + 0.05 * w, y1 + below * h)], fill=255, width=2)
    return out.convert("RGB")


def masks(size, boxes, below=0.8, feather=18):
    """返回（涂色用的硬蒙版，重画用的羽化蒙版）。
    只圈耳朵那一小块，模型看着旁边的下巴、脖子还会把耳朵画回来；
    所以从耳朵往下连成一条，涂成头发色，让它画成一缕垂下来的鬓发。"""
    hard = Image.new("L", size, 0)
    soft = Image.new("L", size, 0)
    dh, ds = ImageDraw.Draw(hard), ImageDraw.Draw(soft)
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        dh.rounded_rectangle([x0 - 0.1 * w, y0 - 0.1 * h, x1 + 0.1 * w, y1 + below * h], radius=w / 2, fill=255)
        ds.rounded_rectangle([x0 - 0.3 * w, y0 - 0.4 * h, x1 + 0.3 * w, y1 + (below + 0.3) * h], radius=w / 2, fill=255)
    # 耳朵小的时候，整块一起羽化，中间到不了全白：重画不彻底，贴回去原来的耳朵还透出来（灰影、白斑）。
    # 所以涂色那一条外扩一点、保证全白，只有外圈羽化。
    core = hard.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(3))
    soft = ImageChops.lighter(soft.filter(ImageFilter.GaussianBlur(feather)), core)
    return hard.filter(ImageFilter.GaussianBlur(3)), soft


# 力度 0.9 时模型会把耳朵重新画出来；0.6 留住涂上的头发色，配密发丝线最干净（2026-11-03 试过 0.75、0.6）
def fix(src, out_dir, model, lora, seed=9902, denoise=0.6, cn_strength=1.0):
    im = Image.open(src).convert("RGB")
    faces = model(str(src), verbose=False)[0].boxes
    if not len(faces):
        return None
    face = max(faces.xyxy.tolist(), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    boxes = ear_boxes(im, face)
    if not boxes:
        return None
    shown = im.copy()
    d = ImageDraw.Draw(shown)
    d.rectangle(face, outline=(0, 160, 255), width=3)
    for b in boxes:
        d.rectangle(b, outline=(255, 0, 0), width=4)
    shown.save(out_dir / f"{src.stem}_遮耳框.png")

    hard, soft = masks(im.size, boxes)
    painted = Image.composite(Image.new("RGB", im.size, hair_color(im, hard)), im, hard)
    img_name = upload(painted, f"ze_{src.stem}.png")
    mask_name = upload(soft.convert("RGB"), f"ze_{src.stem}_mask.png")
    edges = control_lines(im, hard, boxes)
    edges.save(out_dir / f"{src.stem}_线稿.png")
    edge_name = upload(edges, f"ze_{src.stem}_edge.png")
    wf = build(img_name, mask_name, PROMPT, NEGATIVE + ", " + NEG, seed, 28, 4.5, 1, denoise,
               f"zhou_comic/遮耳朵_{src.stem}", lora, 0.8)
    wf["20"] = {"class_type": "LoadImage", "inputs": {"image": edge_name}}
    wf["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROLNET}}
    wf["22"] = {"class_type": "SetUnionControlNetType",
                "inputs": {"control_net": ["21", 0], "type": "canny/lineart/anime_lineart/mlsd"}}
    wf["23"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["2", 0], "negative": ["3", 0], "control_net": ["22", 0], "image": ["20", 0],
        "strength": cn_strength, "start_percent": 0.0, "end_percent": 0.8, "vae": ["1", 2]}}
    wf["5"]["inputs"]["positive"] = ["23", 0]
    wf["5"]["inputs"]["negative"] = ["23", 1]
    # Differential Diffusion：蒙版的灰度就是重画的力度
    wf["9"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": wf["5"]["inputs"]["model"]}}
    wf["5"]["inputs"]["model"] = ["9", 0]
    info = run(wf)[0]
    q = urllib.parse.urlencode({"filename": info["filename"], "subfolder": info["subfolder"], "type": info["type"]})
    gen = Image.open(io.BytesIO(get(f"/view?{q}"))).convert("RGB").resize(im.size)
    out = Image.composite(gen, im, soft)
    path = out_dir / f"{src.stem}_遮.png"
    out.save(path)
    # 放大对比：每只耳朵前后各一块，检查有没有接缝、残迹
    pairs = []
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        area = (int(max(0, x0 - w)), int(max(0, y0 - h)), int(min(im.width, x1 + w)), int(min(im.height, y1 + h)))
        pairs += [im.crop(area).resize((300, 300)), out.crop(area).resize((300, 300))]
    sheet = Image.new("RGB", (300 * len(pairs), 300), "white")
    for i, c in enumerate(pairs):
        sheet.paste(c, (300 * i, 0))
    sheet.save(out_dir / f"{src.stem}_放大.png")
    return path


def main():
    from ultralytics import YOLO

    args = sys.argv[1:]
    lora = None
    if args[0] == "--lora":
        lora, args = args[1], args[2:]
    out_dir = Path(args[0]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(FACE_MODEL)
    for src in map(Path, args[1:]):
        t0 = time.time()
        path = fix(src.resolve(), out_dir, model, lora)
        print(src.name, f"{time.time() - t0:.0f} 秒" if path else "没找到耳朵，跳过", flush=True)


if __name__ == "__main__":
    main()

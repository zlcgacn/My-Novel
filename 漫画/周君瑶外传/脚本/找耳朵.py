r"""用 Grounding DINO 按文字“ear”找图里露出来的耳朵。用 ComfyUI 自带的 Python 跑。

用法：
    python_embeded\python.exe 找耳朵.py 输出目录 图1.png 图2.png ...
在输出目录里给每张图画出耳朵框（<原名>_耳朵框.png），并把框写进 输出目录/_耳朵.json：{原图路径: [[x0,y0,x1,y1], ...]}。
模型在 D:\AI\models\grounding-dino-tiny（2026-11-03 下载，约 690MB）。
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL = r"D:\AI\models\grounding-dino-tiny"
THRESHOLD = 0.25

_proc = _model = None


def find_ears(im):
    """返回 [(框, 分数), ...]，框是 [x0, y0, x1, y1]。"""
    global _proc, _model
    if _model is None:
        _proc = AutoProcessor.from_pretrained(MODEL)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL)
    im = im.convert("RGB")
    inputs = _proc(images=im, text="an ear.", return_tensors="pt")
    with torch.no_grad():
        out = _model(**inputs)
    res = _proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=THRESHOLD, text_threshold=THRESHOLD, target_sizes=[im.size[::-1]])[0]
    ears = []
    for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
        w, h = box[2] - box[0], box[3] - box[1]
        # 太大的框是整个头
        if w * h > 0.06 * im.width * im.height:
            continue
        ears.append(([round(v) for v in box], round(score, 2)))
    return ears


def main():
    out_dir = Path(sys.argv[1]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    for src in map(Path, sys.argv[2:]):
        src = src.resolve()
        im = Image.open(src).convert("RGB")
        ears = find_ears(im)
        found[str(src)] = [b for b, _ in ears]
        d = ImageDraw.Draw(im)
        for b, s in ears:
            d.rectangle(b, outline=(255, 0, 0), width=4)
            d.text((b[0], b[1] - 12), str(s), fill=(255, 0, 0))
        im.save(out_dir / f"{src.stem}_耳朵框.png")
        print(src.name, [s for _, s in ears])
    (out_dir / "_耳朵.json").write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""只重画衣服，脸和头发原样保留。用 ComfyUI 自带的 Python 跑（要 Pillow）。

用法：
    python_embeded\\python.exe inpaint.py 任务.json

任务文件格式（路径相对任务文件所在目录）：
{
  "out_dir": "../../图/设定/校服标准",
  "jobs": [
    {"name": "zhou_portrait_uni", "image": "../../图/设定/定稿/便服/V1_zhou_s1101_2.png",
     "prompt": "...", "seed": 7001, "batch": 4, "denoise": 0.9}
  ]
}

衣服的范围按颜色自动圈：参考图里便服是绿色，绿色像素就是要重画的地方；
头发是黑的、脸和领子是白的，饱和度低，不会被圈进去。
出图后再把圈外的原图像素贴回去，保证脸和头发一个像素都不变。

方框模式（多人同框时给某一个人换脸）：job 里写 "box": [x0, y0, x1, y1]（原图像素），
只重画框里的部分。框连同四周一圈裁出来放大到 1024 再画，细节才够，画完缩回去贴上。
可以加 "lora"、"lora_strength" 挂角色 LoRA。
"""

import io
import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from comfy import CKPT, NEGATIVE, POSITIVE_HEAD, get, post, upload  # noqa: E402


def clothes_mask(im, grow=4, feather=3):
    """绿色、有一定饱和度的像素算衣服。返回 L 模式蒙版，255 = 重画。"""
    hsv = im.convert("HSV")
    h, s, v = hsv.split()
    # PIL 的色相 0-255；绿色大约 40-140
    px_h, px_s, px_v = h.load(), s.load(), v.load()
    m = Image.new("L", im.size, 0)
    pm = m.load()
    w, ht = im.size
    for y in range(ht):
        for x in range(w):
            if 40 <= px_h[x, y] <= 140 and px_s[x, y] >= 45 and px_v[x, y] >= 25:
                pm[x, y] = 255
    # 去掉零星噪点，再向外长一点，盖住衣服边缘
    m = m.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.MaxFilter(grow * 2 + 1))
    return m, m.filter(ImageFilter.GaussianBlur(feather))


def box_crop(im, box, pad=0.35, side=1024, feather=24):
    """框四周多裁一圈当上下文，放大到长边 side。返回裁图、蒙版、裁剪范围。"""
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    cx0, cy0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    cx1, cy1 = min(im.width, int(x1 + px)), min(im.height, int(y1 + py))
    k = side / max(cx1 - cx0, cy1 - cy0)
    size = (round((cx1 - cx0) * k / 8) * 8, round((cy1 - cy0) * k / 8) * 8)
    crop = im.crop((cx0, cy0, cx1, cy1)).resize(size, Image.LANCZOS)
    m = Image.new("L", size, 0)
    m.paste(255, (round((x0 - cx0) * k), round((y0 - cy0) * k), round((x1 - cx0) * k), round((y1 - cy0) * k)))
    return crop, m, m.filter(ImageFilter.GaussianBlur(feather)), (cx0, cy0, cx1, cy1)


def build(image_name, mask_name, prompt, negative, seed, steps, cfg, batch, denoise, prefix,
          lora=None, lora_strength=0.8):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": POSITIVE_HEAD + prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "11": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["12", 0], "mask": ["11", 0]}},
        "14": {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["13", 0], "amount": batch}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["14", 0],
                "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": denoise,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }
    if lora:
        wf["8"] = {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": lora,
                                                         "strength_model": lora_strength, "strength_clip": lora_strength}}
        wf["2"]["inputs"]["clip"] = ["8", 1]
        wf["3"]["inputs"]["clip"] = ["8", 1]
        wf["5"]["inputs"]["model"] = ["8", 0]
    return wf


def run(workflow):
    prompt_id = post("/prompt", {"prompt": workflow, "client_id": uuid.uuid4().hex})["prompt_id"]
    while True:
        hist = json.loads(get(f"/history/{prompt_id}"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(entry["status"].get("messages"), ensure_ascii=False))
            return [img for node in entry["outputs"].values() for img in node.get("images", [])]
        time.sleep(1)


def main():
    task_path = Path(sys.argv[1]).resolve()
    task = json.loads(task_path.read_text(encoding="utf-8"))
    base = task_path.parent
    out_dir = (base / task["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for job in task["jobs"]:
        full = Image.open(base / job["image"]).convert("RGB")
        if job.get("box"):
            src, hard, soft, area = box_crop(full, job["box"])
        else:
            src, area = full, None
            hard, soft = clothes_mask(src, job.get("grow", 4))
        hard.save(out_dir / f"{job['name']}_mask.png")
        img_name = upload(src, f"zc_{job['name']}_src.png")
        mask_name = upload(hard.convert("RGB"), f"zc_{job['name']}_mask.png")
        negative = NEGATIVE + (", " + job["negative_extra"] if job.get("negative_extra") else "")
        wf = build(img_name, mask_name, job["prompt"], negative, job["seed"], job.get("steps", 28),
                   job.get("cfg", 4.5), job.get("batch", 4), job.get("denoise", 0.9), f"zhou_comic/{job['name']}",
                   job.get("lora"), job.get("lora_strength", 0.8))
        t0 = time.time()
        for i, info in enumerate(run(wf)):
            q = urllib.parse.urlencode({"filename": info["filename"], "subfolder": info["subfolder"], "type": info["type"]})
            gen = Image.open(io.BytesIO(get(f"/view?{q}"))).convert("RGB").resize(src.size)
            # 圈外贴回原图：脸和头发保证不变
            out = Image.composite(gen, src, soft)
            if area:
                patch = out.resize((area[2] - area[0], area[3] - area[1]), Image.LANCZOS)
                pmask = soft.resize(patch.size, Image.LANCZOS)
                out = full.copy()
                out.paste(patch, area[:2], pmask)
            out.save(out_dir / f"{job['name']}_s{job['seed']}_{i + 1}.png")
        print(f"{job['name']}: {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()

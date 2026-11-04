"""通过 ComfyUI 本地接口出图。只用标准库。

用法：
    python comfy.py 任务.json

任务文件格式：
{
  "out_dir": "../图/设定/候选",        // 相对任务文件所在目录
  "defaults": {"width": 832, "height": 1216, "steps": 28, "cfg": 5.5, "batch": 4},
  "jobs": [
    {"name": "zhou_sheet", "prompt": "...", "seed": 1001, "batch": 4}
  ]
}
每个 job 的 prompt 前面自动加上通用正面开头，负面提示词默认用通用那段，可以用 "negative" 覆盖。

固定身材和脸（2026-11-02 加）：
  "pose": "骨架图路径"   用 骨架.py 画的 OpenPose 骨架控制姿势和头身比（Union ControlNet），
                         "cn_strength" 默认 0.8，"cn_end" 默认 0.8（最后两成步数放开，细节自然些）
  "face_fix": true       出图后把脸裁出来放大重画一遍（FaceDetailer），远景小脸的眼睛也按近景标准画，
                         "face_denoise" 默认 0.4；"face_prompt" 是修脸单用的正面词（写眼睛长什么样），
                         不写就用整张图的提示词；"face_negative" 是修脸单用的负面词（完整一段，不追加）
"""

import io
import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
CKPT = "waiIllustriousSDXL_v170.safetensors"
CONTROLNET = "controlnet-union-sdxl-promax.safetensors"
POSITIVE_HEAD = "masterpiece, best quality, amazing quality, "
# 模型写不了汉字，画面里一个字都不要，字全部由排版脚本加
NEGATIVE = (
    "bad quality, worst quality, worst detail, censor, lowres, bad anatomy, bad hands, "
    "extra fingers, missing fingers, (text:1.3), (chinese text:1.3), (japanese text:1.3), "
    "english text, letters, character name, labels, annotations, color palette, color swatch, "
    "speech bubble, logo, watermark, signature, artist name, nsfw, "
    "cleavage, large breasts, mature female, serafuku, sailor collar, kimono, "
    "school uniform, smartphone, modern clothes, zipper, sneakers, track jacket, "
    "puffer jacket, leather jacket, hoodie, cargo pants, jeans, patch, badge, "
    # 去油光、去写实：AI 感主要来自这几样
    "shiny skin, shiny hair, glossy, oily skin, specular highlights, hair highlights, 3d, "
    "realistic, photorealistic, semi-realistic, airbrush, glowing, bloom, depth of field, "
    "lens flare, sparkle, "
    # 深绿校服会把黑头发带成墨绿
    "green hair, dark green hair, aqua hair"
)


def build_workflow(prompt, negative, seed, width, height, steps, cfg, batch, prefix, head=POSITIVE_HEAD,
                   lora=None, lora_strength=0.8, pose=None, cn_strength=0.8, cn_end=0.8,
                   face_fix=False, face_denoise=0.4, face_prompt=None, face_negative=None):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": head + prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": batch}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
                "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }
    if lora:
        # 角色 LoRA 挂在模型和文本编码器上，固定脸和发型
        wf["8"] = {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": lora,
                                                         "strength_model": lora_strength, "strength_clip": lora_strength}}
        wf["2"]["inputs"]["clip"] = ["8", 1]
        wf["3"]["inputs"]["clip"] = ["8", 1]
        wf["5"]["inputs"]["model"] = ["8", 0]
    model, clip = (["8", 0], ["8", 1]) if lora else (["1", 0], ["1", 1])
    if pose:
        # 骨架只管姿势和头身比，不管画风
        wf["20"] = {"class_type": "LoadImage", "inputs": {"image": pose}}
        wf["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROLNET}}
        wf["22"] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["21", 0], "type": "openpose"}}
        wf["23"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["2", 0], "negative": ["3", 0], "control_net": ["22", 0], "image": ["20", 0],
            "strength": cn_strength, "start_percent": 0.0, "end_percent": cn_end, "vae": ["1", 2]}}
        wf["5"]["inputs"]["positive"] = ["23", 0]
        wf["5"]["inputs"]["negative"] = ["23", 1]
    if face_fix:
        face_pos = ["2", 0]
        if face_prompt:
            wf["32"] = {"class_type": "CLIPTextEncode", "inputs": {"text": head + face_prompt, "clip": clip}}
            face_pos = ["32", 0]
        face_neg = ["3", 0]
        if face_negative:
            # 整张图的负面压了 blush 等去光影的词，会把眼下那点粉红也压掉，修脸另用一份
            wf["33"] = {"class_type": "CLIPTextEncode", "inputs": {"text": face_negative, "clip": clip}}
            face_neg = ["33", 0]
        wf["30"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/face_yolov8m.pt"}}
        wf["31"] = {"class_type": "FaceDetailer", "inputs": {
            "image": ["6", 0], "model": model, "clip": clip, "vae": ["1", 2],
            "guide_size": 512, "guide_size_for": True, "max_size": 1024, "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler_ancestral", "scheduler": "normal", "positive": face_pos, "negative": face_neg,
            "denoise": face_denoise, "feather": 5, "noise_mask": True, "force_inpaint": True,
            "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3.0,
            "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93, "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "False", "drop_size": 10,
            "bbox_detector": ["30", 0], "wildcard": "", "cycle": 1}}
        wf["7"]["inputs"]["images"] = ["31", 0]
    return wf


def post(path, payload):
    req = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(SERVER + path) as r:
        return r.read()


def upload(img, name):
    """把 PIL 图传给 ComfyUI，返回它在 input 目录里的文件名。"""
    buf = io.BytesIO()
    img.save(buf, "PNG")
    boundary = uuid.uuid4().hex
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            'Content-Type: image/png\r\n\r\n')
    tail = (f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue'
            f'\r\n--{boundary}--\r\n')
    body = head.encode() + buf.getvalue() + tail.encode()
    req = urllib.request.Request(SERVER + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["name"]


def run_job(workflow, client_id):
    prompt_id = post("/prompt", {"prompt": workflow, "client_id": client_id})["prompt_id"]
    while True:
        hist = json.loads(get(f"/history/{prompt_id}"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status.get("messages"), ensure_ascii=False))
            images = []
            for node in entry["outputs"].values():
                images += node.get("images", [])
            return images
        time.sleep(1)


def main():
    task_path = Path(sys.argv[1]).resolve()
    task = json.loads(task_path.read_text(encoding="utf-8"))
    out_dir = (task_path.parent / task["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    d = {"width": 832, "height": 1216, "steps": 28, "cfg": 4.5, "batch": 4, **task.get("defaults", {})}
    client_id = uuid.uuid4().hex

    for job in task["jobs"]:
        p = {**d, **job}
        negative = p.get("negative", NEGATIVE)
        if p.get("negative_extra"):
            negative += ", " + p["negative_extra"]
        pose = None
        if p.get("pose"):
            from PIL import Image
            pose = upload(Image.open(task_path.parent / p["pose"]).convert("RGB"), f"zc_pose_{p['name']}.png")
        wf = build_workflow(p["prompt"], negative, p["seed"], p["width"], p["height"],
                            p["steps"], p["cfg"], p["batch"], f"zhou_comic/{p['name']}",
                            p.get("head", POSITIVE_HEAD), p.get("lora"), p.get("lora_strength", 0.8),
                            pose, p.get("cn_strength", 0.8), p.get("cn_end", 0.8),
                            p.get("face_fix", False), p.get("face_denoise", 0.4), p.get("face_prompt"),
                            p.get("face_negative"))
        t0 = time.time()
        images = run_job(wf, client_id)
        for i, img in enumerate(images):
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
            dest = out_dir / f"{p['name']}_s{p['seed']}_{i + 1}.png"
            dest.write_bytes(get(f"/view?{q}"))
            print(dest.name)
        print(f"  {p['name']}: {len(images)} 张，{time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()

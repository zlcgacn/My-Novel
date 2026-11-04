r"""周君瑶 LoRA 稳定性测试：固定 13 个场景、固定种子，各出 4 张，拼成总览，用来比较各版 LoRA。

用法：
    python_embeded\python.exe 稳定性测试.py LoRA文件名 轮次名 [强度]
    例：稳定性测试.py zhou_junyao_e.safetensors r1

出图在 图/稳定性测试/<轮次名>/，总览是同目录的 总览-1.png、总览-2.png。
每张图检查五项：脸、五官（眼睛）、头发（含耳朵）、身高、比例，结果记在 稳定性测试记录.md。
提示词就是 角色设定.md 里的全套写法：E 画风、骨架、修脸单写眼睛和头发、盖耳朵。
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import comfy  # noqa: E402

HERE = Path(__file__).parent
SKEL = "../../图/骨架/"

# 2026-11-03 第 4 轮起退回 r3 的写法：第 3 轮加的 (long sidelocks:1.4)、hair curtain、thick hair in front of ears
# 画成脸前挂几根硬黑条，候选图带着这毛病进了素材，e4 整版作废。耳朵改用 遮耳朵.py 事后盖。
# 压手部动作（写 covering ears 会画成捂耳朵）保留。压紫发、蓝发的负面去掉：它把整张图往绿推，头发发绿、皮肤发黄。
HAIR = "(hair over ears:1.3), (sidelocks:1.2), hair framing face, "
HEAD = ("zhoujunyao, 1girl, solo, child, (black hair:1.2), long hair, straight hair, " + HAIR +
        "flat color, (no shading:1.2), flat lighting, matte, ")
FULL = ("(dark green dress:1.3), (long dress:1.4), (ankle-length skirt:1.3), long sleeves, white collar, high collar, "
        "black footwear, ankle boots, ")
UPPER = "dark green dress, long sleeves, white collar, high collar, "  # 加权会整张偏绿
FACE = "zhoujunyao, 1girl, solo, child, (black hair:1.2), long hair, blunt bangs, " + HAIR + "face, small face, "
EYES = ("(dark brown eyes:1.3), (black pupils:1.2), (large eyes:1.1), (almond eyes:1.2), (jitome:0.5), "
        "upper eyelid slightly covering top of iris, (thick black upper eyeliner:1.2), eyelashes at outer corner, "
        "(pink lower eyelid:1.3), (reddish lower lash line:1.3), faint pink under eyes, ")
# e7 起素材标注区分了 hair over ears / ears visible through hair，负面压后者
EARS = ("(ears visible through hair:1.3), (ears:1.3), visible ears, hair behind ear, hair tucked behind ears, short hair, bob cut, "
        "(hands on own ears:1.4), (covering ears:1.4), hand on head, hand in hair, hands up")
NOLIGHT = ("(reflection:1.2), reflective, reflected light, (rim lighting:1.2), backlighting, (specular highlights:1.3), "
           "hair highlights, shiny, sheen, glint, light reflection, light rays, sunbeam, light particles, "
           "dramatic lighting, cinematic lighting, volumetric lighting, (cast shadow:1.2), strong shadows, high contrast, "
           "chiaroscuro, gradient, soft shading, ambient occlusion, shiny footwear, patent leather, glossy footwear, "
           "light glow, glowing lamp, lamplight, warm lighting, blush, (green tint:1.2), color cast")
# 全身的裙子会往灰蓝跑（r3、r4 都有），压住；近景不能压，压了整张偏绿
FULL_NEG = (", olive, olive green, khaki, light green dress, grey dress, (blue dress:1.2), (grey-blue dress:1.2), navy, "
            "(knee-length:1.3), (bare legs:1.2), thighs")
UPPER_NEG = ", (full body:1.4), feet, footwear, boots, legs, shoes"
FACE_NEG = (comfy.NEGATIVE + ", (wide-eyed:1.3), (round eyes:1.2), staring, glaring, (dark circles:1.3), (eye bags:1.2), "
            "black lower eyelashes, dark lower lash line, heavy makeup, shiny skin, glossy, " + EARS)

# 名字, 种子, 景别(full/upper), 骨架, 画面, 整张图的表情, 修脸的表情（None 表示不修脸：背影看不到脸）
SHOTS = [
    ("01_近景正面", 20001, "upper", None, "(portrait:1.3), close-up, looking at viewer, white background",
     "expressionless", "expressionless, empty eyes"),
    ("02_近景左侧", 20011, "upper", None, "(portrait:1.3), close-up, three quarter view, facing left, looking at viewer, white background",
     "expressionless", "expressionless, empty eyes"),
    ("03_近景右侧", 20021, "upper", None, "(portrait:1.3), close-up, three quarter view, facing right, looking to the side, white background",
     "expressionless", "expressionless"),
    ("04_正侧面", 20031, "upper", None, "upper body, (profile:1.3), from side, looking ahead, white background",
     "expressionless", "expressionless, profile"),
    ("05_俯拍", 20041, "upper", None, "upper body, (from above:1.3), looking up at viewer, indoors, wooden floor",
     "expressionless", "looking up, expressionless"),
    ("06_仰拍", 20051, "upper", None, "upper body, (from below:1.3), looking down at viewer, sky, clouds",
     "expressionless, cold", "looking down, expressionless"),
    ("07_大笑", 20061, "upper", None, "upper body, indoors, bedroom", "laughing, closed eyes, open mouth",
     "closed eyes, laughing, open mouth"),
    ("08_生气", 20071, "upper", None, "upper body, looking at viewer, clenched hands, white background",
     "angry, shouting, open mouth, frown", "angry, shouting, open mouth, frown"),
    ("09_全身正面", 20081, "full", "稳定性/full_front.png", "full body, standing, looking at viewer, white background",
     "expressionless", "expressionless, empty eyes"),
    ("10_全身左侧", 20091, "full", "稳定性/full_3q_left.png", "full body, standing, three quarter view, white background",
     "expressionless", "expressionless"),
    ("11_走路", 20101, "full", "稳定性/walk.png", "full body, (walking:1.3), (striding:1.2), mid-step, from side, street, stone pavement, overcast",
     "expressionless", "expressionless"),
    ("12_背影", 20111, "full", "稳定性/back.png", "full body, (from behind:1.4), (back view:1.3), standing, corridor, white wall",
     "", None),
    ("13_坐着写字", 20121, "full", "稳定性/desk.png", "sitting at desk, writing letter, (holding pen:1.2), paper, indoors, from side",
     "expressionless", "looking down, expressionless"),
]


def jobs_for(lora, strength):
    jobs = []
    for name, seed, frame, pose, scene, expr, fexpr in SHOTS:
        neg = NOLIGHT + ", " + EARS + (FULL_NEG if frame == "full" else UPPER_NEG)
        j = {"name": name, "seed": seed, "lora": lora, "lora_strength": strength,
             "prompt": HEAD + (FULL if frame == "full" else UPPER) + scene + (", " + expr if expr else ""),
             "negative_extra": neg}
        if pose:
            j["pose"] = SKEL + pose
        if fexpr is not None:
            eyes = "" if "closed eyes" in fexpr else EYES
            fneg = FACE_NEG if eyes else FACE_NEG.replace("(wide-eyed:1.3), (round eyes:1.2), staring, glaring, ", "")
            j.update(face_fix=True, face_denoise=0.5, face_prompt=FACE + eyes + fexpr, face_negative=fneg)
        jobs.append(j)
    return jobs


def overview(out_dir):
    W, H = 300, 438
    for part, shots in (("1", SHOTS[:7]), ("2", SHOTS[7:])):
        g = Image.new("RGB", (W * 4 + 150, (H + 6) * len(shots)), "white")
        d = ImageDraw.Draw(g)
        for r, (name, seed, *_) in enumerate(shots):
            d.text((6, r * (H + 6) + H // 2), name, fill="black")
            for c in range(4):
                f = out_dir / f"{name}_s{seed}_{c + 1}.png"
                if f.exists():
                    g.paste(Image.open(f).resize((W, H)), (150 + c * W, r * (H + 6)))
        g.save(out_dir / f"总览-{part}.png")


def main():
    lora, tag = sys.argv[1], sys.argv[2]
    strength = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    out_rel = f"../../图/稳定性测试/{tag}"
    task = {"out_dir": out_rel, "defaults": {"batch": 4}, "jobs": jobs_for(lora, strength)}
    task_path = HERE / "任务" / f"稳定性测试-{tag}.json"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.argv = ["comfy.py", str(task_path)]
    comfy.main()
    overview((task_path.parent / out_rel).resolve())
    print("总览已拼好")


if __name__ == "__main__":
    main()

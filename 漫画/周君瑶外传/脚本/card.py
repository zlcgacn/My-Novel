"""把定稿设定图排成一张带中文标注的设定卡。用 ComfyUI 自带的 Python 跑（有 Pillow）。

用法：
    python_embeded\\python.exe card.py 卡片.json

卡片文件格式（路径相对卡片文件所在目录）：
{
  "out": "../../图/设定/定稿/设定卡-周君瑶.png",
  "title": "周君瑶",
  "subtitle": "十二岁 · 中等学校一年级",
  "images": [{"path": "...png", "caption": "三视图"}],
  "notes": ["认法：……", "……"]
}
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = [Path(r"C:\Windows\Fonts"), Path.home() / r"AppData\Local\Microsoft\Windows\Fonts"]
SERIF = "NotoSerifSC-VF.ttf"
SANS = "NotoSansSC-VF.ttf"

INK = (30, 30, 30)
GREY = (110, 110, 110)
RULE = (190, 190, 190)
PAPER = (250, 249, 246)


def font(name, size, weight):
    for d in FONT_DIRS:
        p = d / name
        if p.exists():
            f = ImageFont.truetype(str(p), size)
            try:
                f.set_variation_by_axes([weight])
            except (OSError, ValueError):
                pass
            return f
    raise FileNotFoundError(name)


def main():
    spec_path = Path(sys.argv[1]).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = spec_path.parent

    img_h = spec.get("img_h", 1000)
    gap = 48
    margin = 80
    ims = []
    for item in spec["images"]:
        im = Image.open(base / item["path"]).convert("RGB")
        ims.append((im.resize((round(im.width * img_h / im.height), img_h), Image.LANCZOS), item["caption"]))

    title_f = font(SERIF, 96, 700)
    sub_f = font(SANS, 40, 400)
    cap_f = font(SANS, 36, 500)
    note_f = font(SANS, 38, 400)

    width = margin * 2 + sum(im.width for im, _ in ims) + gap * (len(ims) - 1)
    notes = spec.get("notes", [])
    height = margin + 120 + 70 + 40 + img_h + 80 + len(notes) * 62 + margin

    card = Image.new("RGB", (width, height), PAPER)
    d = ImageDraw.Draw(card)

    y = margin
    d.text((margin, y), spec["title"], font=title_f, fill=INK)
    tw = d.textlength(spec["title"], font=title_f)
    d.text((margin + tw + 36, y + 46), spec.get("subtitle", ""), font=sub_f, fill=GREY)
    y += 120 + 40
    d.line([(margin, y), (width - margin, y)], fill=RULE, width=3)
    y += 40

    x = margin
    for im, caption in ims:
        card.paste(im, (x, y))
        cw = d.textlength(caption, font=cap_f)
        d.text((x + (im.width - cw) / 2, y + img_h + 18), caption, font=cap_f, fill=GREY)
        x += im.width + gap
    y += img_h + 80 + 30

    for line in notes:
        d.text((margin, y), line, font=note_f, fill=INK)
        y += 62

    out = (base / spec["out"]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    card.save(out)
    print(out)


if __name__ == "__main__":
    main()

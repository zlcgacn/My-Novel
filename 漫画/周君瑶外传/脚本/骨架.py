r"""按身高表画 OpenPose 骨架图，交给 ControlNet 固定头身比和身高。用 ComfyUI 自带的 Python 跑。

用法：
    python_embeded\python.exe 骨架.py 骨架.json

骨架文件格式（路径相对骨架文件所在目录）：
{
  "out_dir": "../../图/骨架",
  "frames": [
    {"name": "stand_front", "width": 832, "height": 1216,
     "figures": [{"who": "周君瑶", "pose": "stand_front", "x": 416, "y": 110, "head": 160}]}
  ]
}
x 是脖子（两肩中点）的横坐标，y 是头顶的纵坐标，head 是一个头高的像素。
同一格里几个人时，head 按身高表折算：谁高谁矮由 BODY 里的身高和头身比决定，不要每个人单独给。
画面外的关节照样算，只是画不出来，ControlNet 能处理半身构图。

python_embeded\python.exe 骨架.py --lineup 输出.png  画全员身高对照（带名字，只给人看，不喂 ControlNet）。
"""

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 身高（厘米）、头身比、肩宽、胯宽、脖子长（后三项是相对标准体型的倍数）
BODY = {
    "周君瑶": (145, 6.0, 1.00, 1.00, 1.00),
    "早川千鹤": (140, 6.0, 0.95, 0.95, 1.00),
    "同桌": (143, 6.0, 1.00, 1.00, 1.00),
    "鹅": (150, 6.2, 1.00, 0.95, 1.40),
    "圆脸": (147, 5.8, 1.25, 1.35, 0.70),
    "姜振宇": (160, 6.5, 1.05, 0.95, 1.00),
    "级任先生": (183, 7.5, 1.30, 1.05, 1.00),
    "校长": (165, 7.0, 1.30, 1.25, 0.90),
    "老护院": (178, 7.0, 1.50, 1.25, 0.80),
    "管家": (170, 7.2, 1.20, 1.00, 1.00),
}

# 标准体型（6 头身）各段长度，单位：头高。成人头身比更高时，下巴以下按比例拉长。
NECK, SHOULDER, HIP = 0.30, 0.62, 0.36
UPPER_ARM, FOREARM, TORSO, THIGH, SHIN = 0.95, 0.85, 1.70, 1.45, 1.40

# 姿势：各段方向，0 度朝正下，正数往画面右转。
# fs 是透视缩短（朝镜头或背离镜头的那段画短），body_turn / head_turn 是身体和脸朝向，
# -1 朝画面左（看得到她的左侧），0 正面，1 朝画面右（看得到她的右侧）。
# back=True 是背影：左右对调，脸上只剩两只耳朵。
POSES = {
    "stand_3q_left": dict(torso=0, head=0, body_turn=-0.35, head_turn=-0.45,
                          r_upper=-6, r_fore=-3, l_upper=6, l_fore=3, r_thigh=-4, r_shin=0, l_thigh=3, l_shin=0),
    "walk_3q": dict(torso=2, head=0, body_turn=0.25, head_turn=0.4,
                    r_upper=-14, r_fore=-20, l_upper=12, l_fore=6,
                    r_thigh=-11, r_shin=-3, l_thigh=12, l_shin=6),
    "back_stand": dict(torso=0, head=0, back=True, r_upper=8, r_fore=4, l_upper=-8, l_fore=-4,
                       r_thigh=3, r_shin=0, l_thigh=-3, l_shin=0),
    "stand_front": dict(torso=0, head=0, r_upper=-8, r_fore=-4, l_upper=8, l_fore=4,
                        r_thigh=-3, r_shin=0, l_thigh=3, l_shin=0),
    "stand_look_side": dict(torso=0, head=0, head_turn=0.6, r_upper=-6, r_fore=-3, l_upper=6, l_fore=3,
                            r_thigh=-3, r_shin=0, l_thigh=3, l_shin=0),
    "stand_hold_front": dict(torso=0, head=4, body_turn=0.3, head_turn=0.15,
                             r_upper=-4, r_fore=62, l_upper=6, l_fore=-58,
                             r_thigh=-2, r_shin=0, l_thigh=4, l_shin=1),
    "sit_front": dict(torso=0, head=0, r_upper=-14, r_fore=-30, l_upper=14, l_fore=30,
                      r_thigh=-6, r_shin=-2, l_thigh=6, l_shin=2, fs=dict(r_thigh=0.35, l_thigh=0.35)),
    "sit_desk_right": dict(torso=-14, head=-28, body_turn=0.75, head_turn=0.85,
                           r_upper=25, r_fore=78, l_upper=35, l_fore=85,
                           r_thigh=82, r_shin=-4, l_thigh=78, l_shin=0,
                           fs=dict(r_thigh=0.85, l_thigh=0.85)),
    "kneel_right": dict(torso=-8, head=-45, body_turn=0.9, head_turn=0.9,
                        r_upper=8, r_fore=62, l_upper=14, l_fore=66,
                        r_thigh=72, r_shin=-92, l_thigh=76, l_shin=-90),
}

# OpenPose 18 点：0 鼻 1 颈 2 右肩 3 右肘 4 右腕 5 左肩 6 左肘 7 左腕 8 右胯 9 右膝 10 右踝
# 11 左胯 12 左膝 13 左踝 14 右眼 15 左眼 16 右耳 17 左耳。连线和颜色照 controlnet_aux 的画法。
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
         (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def vec(deg):
    a = math.radians(deg)
    return math.sin(a), math.cos(a)


def keypoints(who, pose, x, y, head):
    """返回 18 个点（不可见的为 None）。head 是这个人自己的头高像素。"""
    _, heads, sh, hp, nk = BODY[who]
    p = POSES[pose]
    s = (heads - 1) / 5  # 下巴以下拉长的倍数
    fs = p.get("fs", {})
    bt, ht = p.get("body_turn", 0), p.get("head_turn", 0)
    w = math.cos(bt * math.pi / 2)  # 身体侧过去，肩宽、胯宽变窄
    flip = -1 if p.get("back") else 1  # 背影时她的右边在画面右边

    def px(v):
        return v * head

    pts = [None] * 18
    neck = (x, y + px(1 + NECK * nk))
    pts[1] = neck
    # 肩线垂直于躯干
    tx, ty = vec(p["torso"])
    lat = (ty, -tx)  # 指向画面右
    for idx, sign in ((2, -flip), (5, flip)):
        pts[idx] = (neck[0] + sign * lat[0] * px(SHOULDER * sh * w), neck[1] + sign * lat[1] * px(SHOULDER * sh * w))
    hipmid = (neck[0] + tx * px(TORSO * s), neck[1] + ty * px(TORSO * s))
    for idx, sign in ((8, -flip), (11, flip)):
        pts[idx] = (hipmid[0] + sign * lat[0] * px(HIP * hp * w), hipmid[1] + sign * lat[1] * px(HIP * hp * w))

    def chain(start, a, b, la, lb, ka, kb):
        dx, dy = vec(p[ka])
        m = (pts[start][0] + dx * px(la * s * fs.get(ka, 1)), pts[start][1] + dy * px(la * s * fs.get(ka, 1)))
        dx, dy = vec(p[kb])
        e = (m[0] + dx * px(lb * s * fs.get(kb, 1)), m[1] + dy * px(lb * s * fs.get(kb, 1)))
        pts[a], pts[b] = m, e

    chain(2, 3, 4, UPPER_ARM, FOREARM, "r_upper", "r_fore")
    chain(5, 6, 7, UPPER_ARM, FOREARM, "l_upper", "l_fore")
    chain(8, 9, 10, THIGH, SHIN, "r_thigh", "r_shin")
    chain(11, 12, 13, THIGH, SHIN, "l_thigh", "l_shin")

    # 头：沿“头的上方向”量，低头就往前倾
    ux, uy = vec(180 + p["head"])
    hl = (-uy, ux)  # 头的画面右方向

    def head_pt(up, side):
        return (neck[0] + ux * px(up) + hl[0] * px(side), neck[1] + uy * px(up) + hl[1] * px(side))

    # 从头顶量：眼睛 0.5 头、耳朵 0.55 头、鼻尖 0.65 头；换算成从肩线往上量
    cw = math.cos(ht * math.pi / 2)
    nkl = NECK * nk
    pts[0] = head_pt(0.35 + nkl, 0.28 * ht)
    pts[14] = head_pt(0.5 + nkl, 0.17 * ht - 0.17 * cw)
    pts[15] = head_pt(0.5 + nkl, 0.17 * ht + 0.17 * cw)
    pts[16] = head_pt(0.45 + nkl, -0.36 * cw - 0.1 * ht)
    pts[17] = head_pt(0.45 + nkl, 0.36 * cw - 0.1 * ht)
    # 侧脸时远侧的眼睛、朝前那侧的耳朵看不见
    if ht > 0.65:
        pts[15] = None
    if ht < -0.65:
        pts[14] = None
    if ht > 0.3:
        pts[17] = None
    if ht < -0.3:
        pts[16] = None
    if flip < 0:
        pts[0] = pts[14] = pts[15] = None
        pts[16] = head_pt(0.45 + nkl, 0.36)
        pts[17] = head_pt(0.45 + nkl, -0.36)
    return pts


def draw(canvas, pts, scale=1.0):
    d = ImageDraw.Draw(canvas)
    stick = max(2, round(4 * scale))
    for i, (a, b) in enumerate(LIMBS):
        if pts[a] is None or pts[b] is None:
            continue
        (x1, y1), (x2, y2) = pts[a], pts[b]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        length = math.hypot(x2 - x1, y2 - y1) / 2
        ang = math.atan2(y2 - y1, x2 - x1)
        poly = []
        for t in range(0, 360, 15):
            r = math.radians(t)
            ex, ey = length * math.cos(r), stick * math.sin(r)
            poly.append((mx + ex * math.cos(ang) - ey * math.sin(ang), my + ex * math.sin(ang) + ey * math.cos(ang)))
        d.polygon(poly, fill=tuple(int(c * 0.6) for c in COLORS[i]))
    for i, pt in enumerate(pts):
        if pt is not None:
            r = max(2, round(4 * scale))
            d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=COLORS[i])


def head_px(who, ref_who, ref_head):
    """同一格里，按身高表把参照人物的头高折算成这个人的头高。"""
    h, n = BODY[who][:2]
    rh, rn = BODY[ref_who][:2]
    return ref_head * (h / n) / (rh / rn)


def render(spec, base):
    out_dir = (base / spec["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fr in spec["frames"]:
        img = Image.new("RGB", (fr.get("width", 832), fr.get("height", 1216)), "black")
        ref = fr["figures"][0]
        for f in fr["figures"]:
            h = f.get("head") or head_px(f["who"], ref["who"], ref["head"])
            # 同一格里站在同一地面上：脚底对齐参照人物
            y = f.get("y")
            if y is None:
                ground = ref["y"] + ref["head"] * BODY[ref["who"]][1]
                y = ground - h * BODY[f["who"]][1]
            draw(img, keypoints(f["who"], f["pose"], f["x"], y, h), h / 160)
        img.save(out_dir / f"{fr['name']}.png")
        print(out_dir / f"{fr['name']}.png")


def lineup(out):
    names = list(BODY)
    tallest = max(BODY[n][0] for n in names)
    H, margin, col = 1400, 120, 260
    img = Image.new("RGB", (col * len(names) + margin * 2, H + 260), "black")
    ground = margin + H
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 34)
    small = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 26)
    d = ImageDraw.Draw(img)
    for cm in range(0, tallest + 1, 20):
        yy = ground - H * cm / tallest
        d.line([(margin - 40, yy), (img.width - margin + 40, yy)], fill=(60, 60, 60), width=2)
        d.text((10, yy - 16), f"{cm}", font=small, fill=(150, 150, 150))
    for i, n in enumerate(names):
        h, heads = BODY[n][:2]
        hp = H * h / tallest / heads
        x = margin + col * i + col // 2
        draw(img, keypoints(n, "stand_front", x, ground - hp * heads, hp), hp / 160)
        for k in range(1, int(heads) + 1):  # 头身比刻度
            yy = ground - hp * heads + hp * k
            d.line([(x + 95, yy), (x + 115, yy)], fill=(120, 120, 120), width=2)
        tw = d.textlength(n, font=font)
        d.text((x - tw / 2, ground + 30), n, font=font, fill=(235, 235, 235))
        sub = f"{h}cm · {heads:g} 头身"
        tw = d.textlength(sub, font=small)
        d.text((x - tw / 2, ground + 80), sub, font=small, fill=(160, 160, 160))
    img.save(out)
    print(out)


def main():
    if sys.argv[1] == "--lineup":
        lineup(sys.argv[2])
        return
    spec_path = Path(sys.argv[1]).resolve()
    render(json.loads(spec_path.read_text(encoding="utf-8")), spec_path.parent)


if __name__ == "__main__":
    main()

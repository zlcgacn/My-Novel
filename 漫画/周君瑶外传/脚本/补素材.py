r"""按稳定性测试里的场景写法，换种子多出几组，当训练素材候选。用 ComfyUI 自带的 Python 跑。

用法：
    python_embeded\python.exe 补素材.py LoRA文件名 轮次名 场景编号:组数 ...
    例：补素材.py zhou_junyao_e.safetensors k1 02:3 03:3 04:2 12:1

每组 4 张，种子从测试种子往后错开（不和测试撞种子）。出图在 图/训练素材/候选-<轮次名>/。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfy  # noqa: E402
import 稳定性测试 as t  # noqa: E402


def main():
    lora, tag, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    base = {j["name"][:2]: j for j in t.jobs_for(lora, 0.8)}
    jobs = []
    for spec in specs:
        code, n = spec.split(":")
        for k in range(int(n)):
            j = dict(base[code])
            j["seed"] = j["seed"] + 1000 * (k + 1) + sum(map(ord, tag)) % 997  # 不用 hash()：每次运行结果不同
            j["name"] = f"{j['name']}_{k + 1}"
            jobs.append(j)
    out_rel = f"../../图/训练素材/候选-{tag}"
    task_path = Path(__file__).parent / "任务" / f"补素材-{tag}.json"
    task_path.write_text(json.dumps({"out_dir": out_rel, "defaults": {"batch": 4}, "jobs": jobs},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    sys.argv = ["comfy.py", str(task_path)]
    comfy.main()


if __name__ == "__main__":
    main()

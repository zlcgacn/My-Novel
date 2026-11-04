#!/bin/bash
# 安装 LoRA 训练环境：kohya sd-scripts + 独立 Python 3.11，全部放 D:\AI，不碰 C 盘和系统 Python
set -e
ROOT=/d/AI
export UV_CACHE_DIR="D:\\AI\\uv\\cache"
export UV_PYTHON_INSTALL_DIR="D:\\AI\\uv\\python"
export HF_HOME="D:\\AI\\hf"

/d/Programs/Python314/python.exe -m pip install --quiet --upgrade uv
UV=/d/Programs/Python314/Scripts/uv.exe

[ -d "$ROOT/sd-scripts" ] || git clone --depth 1 https://github.com/kohya-ss/sd-scripts "$ROOT/sd-scripts"
cd "$ROOT/sd-scripts"
git log -1 --format="sd-scripts %h %cd"

[ -d venv ] || "$UV" venv venv --python 3.11
PY="$ROOT/sd-scripts/venv/Scripts/python.exe"

# 50 系显卡要 CUDA 12.8 以上的 PyTorch
"$UV" pip install --python "$PY" torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$PY" -r requirements.txt

"$PY" -c "import torch;print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "安装完成"

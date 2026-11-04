#!/bin/bash
# 训练周君瑶角色 LoRA。触发词 zhoujunyao；标注里不写黑发、齐刘海、眼睛，让 LoRA 把脸和发型学进触发词。
# 用法：bash 训练-周君瑶.sh [素材目录名] [输出名]
#   第一版：bash 训练-周君瑶.sh zhou zhou_junyao          （V1 画风，36 张，一半是作废的校服）
#   第二版：bash 训练-周君瑶.sh zhou_e zhou_junyao_e      （2026-11-02，E 画风，按定稿逐张挑过眼睛和头发）
# 试过用缠红头绳的素材重训，头绳太小学不进去，强度调高还会崩脸；红头绳后来被作者否决。
# 训练集路径带中文，sd-scripts 处理不稳，先复制到纯英文目录。
set -e
export HF_HOME="D:\\AI\\hf"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SET="${1:-zhou}"
NAME="${2:-zhou_junyao}"
SRC="/d/Novels/My-Novel/漫画/周君瑶外传/图/训练素材/$SET"
WORK=/d/AI/train/zhou
[ -d "$SRC" ] || { echo "没有素材目录 $SRC"; exit 1; }
rm -rf "$WORK/data" "$WORK/out" && mkdir -p "$WORK/data" "$WORK/out"
cp -r "$SRC"/* "$WORK/data/"

# 先让 ComfyUI 把模型从显存里卸掉（最好整个关掉：Windows 虚拟内存额度不够时，申请显存也会报“显存不足”）
curl -s -X POST http://127.0.0.1:8188/free -H "Content-Type: application/json" \
     -d '{"unload_models": true, "free_memory": true}' || true

cd /d/AI/sd-scripts
venv/Scripts/python.exe sdxl_train_network.py \
  --pretrained_model_name_or_path "D:\\AI\\ComfyUI_windows_portable\\ComfyUI\\models\\checkpoints\\waiIllustriousSDXL_v170.safetensors" \
  --train_data_dir "D:\\AI\\train\\zhou\\data" \
  --output_dir "D:\\AI\\train\\zhou\\out" --output_name "$NAME" \
  --caption_extension .txt --resolution 1024,1024 \
  --enable_bucket --min_bucket_reso 640 --max_bucket_reso 1280 --bucket_reso_steps 64 \
  --network_module networks.lora --network_dim 16 --network_alpha 8 --network_train_unet_only \
  --learning_rate 1e-4 --optimizer_type AdamW --lr_scheduler cosine --lr_warmup_steps 50 \
  --train_batch_size 1 --max_train_epochs 8 --save_every_n_epochs 2 \
  --mixed_precision bf16 --save_precision bf16 --sdpa --gradient_checkpointing \
  --cache_latents --vae_batch_size 1 --cache_text_encoder_outputs --no_half_vae \
  --lowram --seed 42 --max_data_loader_n_workers 0 --save_model_as safetensors

# 每一轮都放进 loras，测完再删用不上的
cp "$WORK/out/$NAME"*.safetensors "/d/AI/ComfyUI_windows_portable/ComfyUI/models/loras/"
mkdir -p "$WORK/$NAME" && cp "$WORK/out/$NAME"*.safetensors "$WORK/$NAME/"
ls -la "$WORK/out/"
echo "训练完成"

# 第 5 轮：旧素材和 k6 候选盖耳朵。ComfyUI 要先开着（只开一次，不重启、不弹窗）。
# 已经盖好的（有 _放大.png）、确定没耳朵的（有 .skip_ 标记）跳过。
PY=/d/AI/ComfyUI_windows_portable/python_embeded/python.exe
SC="/d/Novels/My-Novel/漫画/周君瑶外传/脚本"
TS="/d/Novels/My-Novel/漫画/周君瑶外传/图/训练素材"
export PYTHONIOENCODING=utf-8
run_dir() {
  out="$1"; shift
  todo=()
  for f in "$@"; do b=$(basename "$f" .png); [ -f "$out/${b}_放大.png" ] || [ -f "$out/.skip_$b" ] || todo+=("$f"); done
  echo "== $out 还剩 ${#todo[@]} 张"
  [ ${#todo[@]} -eq 0 ] && return
  if $PY -X utf8 "$SC/遮耳朵.py" --lora zhou_junyao_e5.safetensors "$out" "${todo[@]}" | grep -v Loading | tee -a "$out/_log.txt"; then
    # 只有整批正常跑完，才给没出结果的打“没耳朵”标记（中途出错的不打，下次重跑）
    if [ "${PIPESTATUS[0]}" = 0 ]; then
      for f in "${todo[@]}"; do b=$(basename "$f" .png); [ -f "$out/${b}_放大.png" ] || touch "$out/.skip_$b"; done
    fi
  fi
}
mkdir -p "$TS/遮耳朵-e5旧" "$TS/遮耳朵-k6"
run_dir "$TS/遮耳朵-e5旧" $(ls "$TS/zhou_e5/3_zhoujunyao/"*.png | grep -v -e "/k5_" -e "/k4_" -e "/f_")
run_dir "$TS/遮耳朵-k6" "$TS/候选-k6/"*.png
echo "== 全部完成"

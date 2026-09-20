#!/bin/bash
# 0.8B 训练夜间看护（确定性脚本，供定时任务以最小开销调用）
# 逻辑：A/B 完成→退出；训练完成→跑 A/B；进程活且推进→退出；卡死/不在→杀掉并续训。
PROJ="/Users/wayne/Desktop/工作文档库/05-网站与AI工作区/初中单词判定器"
PY="/Users/wayne/venvs/mlx/bin/python"
BASE="/Users/wayne/models/Qwen3.5-0.8B-MLX"
AD="$PROJ/adapters/08b"

# 1) A/B 已完成？
AB="$PROJ/data_v4/AB对分_416金标.tsv"
if [ -f "$AB" ] && head -1 "$AB" | grep -q "0.8B_E" && [ "$(wc -l < "$AB" | tr -d ' ')" -gt 2 ]; then
  echo "WATCHDOG: A/B 已完成，无事可做。"; exit 0
fi

# 2) 训练完成？（最新编号检查点 ≥ 003000）
LATEST=$(ls "$AD" 2>/dev/null | grep -E "^00[0-9]+_adapters" | sort | tail -1)
LATESTN=${LATEST%%_*}; LATESTN=${LATESTN#00}; LATESTN=$((10#$LATESTN))
if [ -f "$AD/adapters.safetensors" ] && [ "$LATESTN" -ge 3000 ]; then
  echo "WATCHDOG: 训练完成，启动 A/B 对分……"
  cd "$PROJ" && $PY scripts/ab_test.py > data_v4/AB结果.log 2>&1
  tail -8 data_v4/AB结果.log
  exit 0
fi

# 3) 进程活且推进？
if ps aux | grep "[m]lx_lm.lora" | grep -q "08b"; then
  if [ -n "$LATEST" ]; then
    M=$(stat -f %m "$AD/$LATEST"); NOW=$(date +%s); AGE=$(( (NOW - M) / 60 ))
    if [ "$AGE" -lt 30 ]; then
      echo "WATCHDOG: 训练推进正常（最新检查点 $LATEST，${AGE} 分钟前）。"; exit 0
    fi
    echo "WATCHDOG: 进程在但 ${AGE} 分钟无新检查点，判定卡死，杀掉续训。"
    pkill -f "mlx_lm.lora"; sleep 5
  fi
fi

# 5) （重）启动：有检查点则续训
cd "$PROJ"
if [ -n "$LATEST" ] && [ -f "$AD/$LATEST" ]; then
  echo "WATCHDOG: 从 $LATEST 续训。"
  nohup $PY -m mlx_lm.lora --model "$BASE" --data data_v4 --train --iters 3000 \
    --batch-size 4 --num-layers 16 --learning-rate 1e-4 --grad-checkpoint --mask-prompt \
    --steps-per-report 500 --steps-per-eval 1000 \
    --adapter-path adapters/08b --resume-adapter-file "adapters/08b/$LATEST" \
    > data_v4/08b续训.log 2>&1 &
else
  echo "WATCHDOG: 无检查点，从头训练。"
  nohup $PY -m mlx_lm.lora --model "$BASE" --data data_v4 --train --iters 3000 \
    --batch-size 4 --num-layers 16 --learning-rate 1e-4 --grad-checkpoint --mask-prompt \
    --steps-per-report 500 --steps-per-eval 1000 \
    --adapter-path adapters/08b \
    > data_v4/08b续训.log 2>&1 &
fi
echo "WATCHDOG: 已启动（PID $!），日志 data_v4/08b续训.log。"

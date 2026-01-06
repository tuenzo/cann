#!/bin/bash
# 运行完整实验并监控资源使用情况
# ===============================

# 设置环境变量（使用 CPU 多线程）
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"

# 激活虚拟环境
source venv/bin/activate

# 输出目录
LOG_DIR="logs/monitoring"
mkdir -p $LOG_DIR

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/monitor_${TIMESTAMP}.log"

echo "============================================================"
echo "运行完整实验 + 资源监控"
echo "============================================================"
echo "时间戳: $TIMESTAMP"
echo "日志文件: $LOG_FILE"
echo "============================================================"
echo ""

# 在后台启动资源监控
python monitor_resources.py --interval 2.0 --output "$LOG_FILE" &
MONITOR_PID=$!

echo "📊 资源监控已启动 (PID: $MONITOR_PID)"
echo "日志文件: $LOG_FILE"
echo ""

# 等待监控器启动
sleep 2

# 运行完整实验
echo "🚀 开始运行完整实验..."
echo "============================================================"
echo ""

python scripts/run_fig2.py --n_runs 20 --n_trials 100 --batch_size 10

# 实验结束
EXIT_CODE=$?

# 停止资源监控
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "============================================================"
echo "实验完成！"
echo "============================================================"
echo "退出代码: $EXIT_CODE"
echo "资源监控日志: $LOG_FILE"
echo "============================================================"

# 显示最后几行日志
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "📋 最近几条监控记录:"
    echo "------------------------------------------------------------"
    tail -20 "$LOG_FILE"
fi

exit $EXIT_CODE


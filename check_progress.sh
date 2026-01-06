#!/bin/bash
# 检查实验进度脚本

PID=$(ps aux | grep "python.*run_fig2.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ 没有找到运行中的实验进程"
    exit 1
fi

echo "=========================================="
echo "实验进程状态检查"
echo "=========================================="
echo ""

# 进程基本信息
echo "📊 进程信息:"
ps -p $PID -o pid,etime,pcpu,pmem,cmd --no-headers | awk '{
    printf "  PID: %s\n", $1
    printf "  运行时间: %s\n", $2
    printf "  CPU 使用率: %s%%\n", $3
    printf "  内存使用: %s%%\n", $4
}'

echo ""
echo "💾 系统资源:"
echo "  CPU 配置: $(nproc) 核心 ($(grep -c processor /proc/cpuinfo) 逻辑处理器)"
free -h | grep Mem | awk '{printf "  内存: %s / %s (可用: %s)\n", $3, $2, $7}'
echo "  CPU 总使用率: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1)%"

echo ""
echo "📁 输出文件:"
if [ -d "results/fig2" ]; then
    echo "  输出目录: results/fig2/"
    ls -lh results/fig2/*.png 2>/dev/null | wc -l | awk '{printf "  已生成图片: %d 个\n", $1}'
    ls -lh results/fig2/*.npz 2>/dev/null | wc -l | awk '{printf "  已生成数据: %d 个\n", $1}'
    echo ""
    echo "  最新文件:"
    ls -lht results/fig2/ | head -5 | tail -4 | awk '{printf "    %s %s %s\n", $6, $7, $9}'
else
    echo "  输出目录不存在"
fi

echo ""
echo "⏱️  时间估算:"
echo "  完整实验: 20 runs × 100 trials = 2000 trials"
echo "  每个 trial: ~6.3 秒 (6300ms 总时长)"
echo "  预计总时间: ~210 分钟 (3.5 小时)"
echo ""
echo "💡 提示:"
echo "  - 程序正在正常运行，CPU 使用率高是正常的"
echo "  - JAX 自动使用多核并行（CPU 使用率可能超过 100%）"
echo "  - 系统配置: $(nproc) 核心可用"
echo "  - 完整实验需要较长时间，请耐心等待"
echo "  - 可以使用 Ctrl+C 中断（会丢失当前进度）"
echo "  - 建议使用 nohup 或 screen 在后台运行"


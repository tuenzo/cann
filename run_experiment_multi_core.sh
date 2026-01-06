#!/bin/bash
# 正确的多核并行实验启动脚本
# ==============================

# 设置环境变量（在激活虚拟环境之前）
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=12"
export JAX_PLATFORMS=cpu

echo "============================================================"
echo "多核并行实验启动"
echo "============================================================"
echo ""
echo "环境变量设置:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "  XLA_FLAGS=$XLA_FLAGS"
echo "  JAX_PLATFORMS=$JAX_PLATFORMS"
echo ""

# 激活虚拟环境
source venv/bin/activate

# 验证环境变量
echo "JAX 配置验证:"
python -c "
import os
import jax
print(f'  JAX 版本: {jax.__version__}')
print(f'  JAX 设备: {jax.devices()}')
print(f'  OMP_NUM_THREADS: {os.getenv(\"OMP_NUM_THREADS\")}')
print(f'  MKL_NUM_THREADS: {os.getenv(\"MKL_NUM_THREADS\")}')
print(f'  XLA_FLAGS: {os.getenv(\"XLA_FLAGS\")}')
"

echo ""
echo "============================================================"
echo "开始实验..."
echo "============================================================"
echo ""

# 运行实验
python scripts/run_fig2.py --n_runs 20 --n_trials 100 --batch_size 10


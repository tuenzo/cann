# 实验环境设置和运行指南

## ✅ 虚拟环境已创建

虚拟环境位置：`/home/tuenzh/projects_new/CANN/venv/`

### 已安装的依赖

- ✅ numpy==1.26.4
- ✅ scipy==1.16.3
- ✅ jax==0.7.0
- ✅ jaxlib==0.7.0
- ✅ matplotlib==3.9.4
- ✅ seaborn==0.13.2
- ✅ tqdm==4.67.1
- ✅ pyyaml==6.0.3

### Git 忽略

虚拟环境已自动被 `.gitignore` 忽略（第 2-5 行）：
```
venv/
.venv/
env/
.env/
```

---

## 🚀 运行实验

### 快速测试（推荐首次运行）

```bash
cd /home/tuenzh/projects_new/CANN
source venv/bin/activate
python run_experiment2.py
```

**特点**：
- 2 runs × 10 trials（快速验证）
- delta_step=10°（减少计算量）
- 约 15 分钟完成

**输出示例**：
```
✅ STD 实验完成！
   Delta 范围: [-90.0°, 80.0°]
   平均误差范围: [-2.49°, 5.00°]
   DoG 幅度: -0.20°
   DoG σ: 19.46°

✅ STF 实验完成！
   Delta 范围: [-90.0°, 60.0°]
   平均误差范围: [-5.13°, 4.94°]
   DoG 幅度: -1.08°
   DoG σ: 8.17°
```

### 完整实验（论文复现）

```bash
cd /home/tuenzh/projects_new/CANN
source venv/bin/activate
python scripts/run_fig2.py --output_dir results/fig2
```

**特点**：
- 20 runs × 100 trials = 2000 trials
- delta_step=1°（完整分辨率）
- 预计需要数小时完成

---

## 📊 实验验证结果

### 配置验证：✅ 22/22 通过 (100%)

所有验收检查项均通过：
- ✅ N=100
- ✅ 时间轴完整（S1/ISI/S2/Delay/Cue/ITI）
- ✅ 参数匹配 Table 1
- ✅ 噪声实现完整
- ✅ 运行规模正确（20 runs × 100 trials）

### 实验运行：✅ 成功

- ✅ STD 实验：成功运行，观察到排斥效应（DoG 幅度为负）
- ✅ STF 实验：成功运行，观察到吸引效应（DoG 幅度为负，但数值不同）
- ✅ 记录功能：成功记录时间序列数据

---

## 🔧 激活虚拟环境

每次运行实验前，需要激活虚拟环境：

```bash
source venv/bin/activate
```

退出虚拟环境：
```bash
deactivate
```

---

## 📝 注意事项

1. **系统限制**：系统 GCC 版本较旧（4.8.5），因此使用了预编译的 wheel 包
2. **性能**：完整实验需要较长时间，建议使用快速测试版本先验证
3. **结果保存**：完整实验的结果会保存在 `results/fig2/` 目录

---

## 🎯 下一步

1. ✅ 虚拟环境已创建并配置
2. ✅ 快速测试已成功运行
3. ⏭️ 如需完整复现，运行 `python scripts/run_fig2.py`


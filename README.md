# Serial Dependence CANN Model

基于JAX实现的两层连续吸引子神经网络（CANN）模型，用于研究视觉感知中的序列依赖现象。

## 项目概述

本项目复现了Zhang et al. (NeurIPS 2025) 论文 *"Neural Correlates of Serial Dependence: Synaptic Short-term Plasticity Orchestrates Repulsion and Attraction"* 中提出的神经网络模型。

### 核心发现

序列依赖（Serial Dependence）是指近期感觉历史如何塑造当前感知，产生两种相反的偏差：

- **排斥效应（Repulsion）**：当前感知被近期刺激排斥，通常发生在感觉处理阶段
- **吸引效应（Attraction）**：当前感知被近期刺激吸引，通常发生在后知觉阶段

本模型通过**短期突触可塑性（STP）**机制解释这些效应：

- **短期抑制（STD）**主导 → 排斥效应（神经递质耗竭）
- **短期增强（STF）**主导 → 吸引效应（释放概率增加）

## 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                    两层CANN网络架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │        Higher Layer (PFC - 前额叶皮层)               │   │
│   │           STF主导 → 吸引效应                         │   │
│   │           τf >> τd (τf=5s, τd=0.3s)                 │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        │                                    │
│              ↑ 前馈     │ 反馈 ↓                            │
│              (J_ff)     │ (J_fb)                            │
│                        │                                    │
│   ┌────────────────────┴────────────────────────────────┐   │
│   │         Lower Layer (V1 - 视觉皮层)                  │   │
│   │           STD主导 → 排斥效应                         │   │
│   │           τd >> τf (τd=3s, τf=0.3s)                 │   │
│   └─────────────────────────────────────────────────────┘   │
│                        ↑                                    │
│                   外部输入                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 安装

### 环境要求

- Python >= 3.9
- JAX >= 0.4.20
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0

### 安装步骤

```bash
# 克隆项目
cd /path/to/project

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# GPU支持 (可选，需要CUDA)
pip install jax[cuda12_pip]
```

## 快速开始

### 运行所有实验

```bash
# 复现所有图表
python scripts/run_all.py --output_dir results --n_trials 20

# 仅复现主要图表（跳过补充实验）
python scripts/run_all.py --skip_supplementary
```

### 单独运行特定实验

```bash
# Figure 2: 单层CANN实验
python scripts/run_fig2.py --output_dir results/fig2

# Figure 3: 两层CANN实验
python scripts/run_fig3.py --output_dir results/fig3

# Figure 4: 时间窗口分析
python scripts/run_fig4.py --output_dir results/fig4

# 补充实验
python scripts/run_supplementary.py --output_dir results/supplementary
```

### Python API使用

```python
from src.models import SingleLayerCANN, TwoLayerCANN, CANNParams, TwoLayerParams
from src.decoding import decode_orientation

# 创建单层CANN（STD主导 → 排斥）
params = CANNParams(tau_d=3.0, tau_f=0.3)
model = SingleLayerCANN(params)

# 运行序列依赖实验
model.reset()
model.present_stimulus(theta_s1=60.0, duration=500)  # S1
model.evolve(1000)  # ISI
model.present_stimulus(theta_s2=90.0, duration=500)  # S2

# 解码感知结果
perceived = model.decode(method='pvm')
error = perceived - 90.0  # 调整误差
print(f"感知: {perceived:.1f}°, 误差: {error:.2f}°")
```

## 项目结构

```
CANN/
├── src/
│   ├── models/
│   │   ├── stp.py              # STP动力学 (STD/STF)
│   │   ├── cann.py             # 单层CANN模型
│   │   └── two_layer_cann.py   # 两层CANN模型
│   ├── decoding/
│   │   └── decoders.py         # 群体解码方法 (PVM/COM/ML/Peak)
│   ├── experiments/
│   │   ├── single_layer_exp.py # 单层实验协议
│   │   ├── two_layer_exp.py    # 两层实验协议
│   │   ├── bayesian_analysis.py# 贝叶斯分析框架
│   │   └── supplementary_exp.py# 补充控制实验
│   ├── analysis/
│   │   └── dog_fitting.py      # DoG曲线拟合
│   └── visualization/
│       └── plots.py            # 图表绘制
├── configs/
│   └── params.py               # 参数配置
├── scripts/
│   ├── run_all.py              # 运行所有实验
│   ├── run_fig2.py             # 复现Figure 2
│   ├── run_fig3.py             # 复现Figure 3
│   ├── run_fig4.py             # 复现Figure 4
│   └── run_supplementary.py    # 复现补充图表
├── results/                    # 输出目录
├── requirements.txt
└── README.md
```

## 核心数学公式

### CANN神经元动力学

$$\tau \frac{du(x,t)}{dt} = -u(x,t) + \rho \int dx' J(x,x') r(x',t) + I_{ext}(x,t)$$

### 发放率（除法归一化）

$$r(x,t) = \frac{u(x,t)^2}{1 + k\rho \int u(x',t)^2 dx'}$$

### STP动力学

$$\frac{dx}{dt} = \frac{1-x}{\tau_d} - u \cdot x \cdot r \quad \text{(神经递质恢复)}$$

$$\frac{du}{dt} = \frac{U-u}{\tau_f} + U(1-u)r \quad \text{(释放概率)}$$

**有效突触强度**: $J_{eff} = J \cdot u \cdot x$

## 复现的图表

| 图号 | 内容 | 关键发现 |
|------|------|----------|
| Fig.2 A-C | STD主导单层CANN | 排斥效应 (~-2°) |
| Fig.2 D-F | STF主导单层CANN | 吸引效应 (~+1.5°) |
| Fig.3 C | Within-trial偏差 | 排斥 (STD in lower layer) |
| Fig.3 D | Between-trial偏差 | 吸引 (STF in higher layer) |
| Fig.4 | ISI/ITI时间窗口 | 偏差随间隔衰减 |
| Fig.S5 | 解码方法对比 | 四种方法结果一致 |
| Fig.S6 | 反转层顺序 | 双重吸引效应 |
| Fig.S7 | 参数敏感性 | 效应幅度变化，类型不变 |
| Fig.S8 | 突触异质性 | 轻微影响效应幅度 |

## 参数配置

### 默认网络参数

| 参数 | 符号 | 默认值 | 描述 |
|------|------|--------|------|
| 神经元数量 | N | 180 | 覆盖180°朝向空间 |
| 时间常数 | τ | 1 ms | 膜电位时间常数 |
| 连接强度 | J₀ | 0.5 | 递归连接 |
| 调谐宽度 | a | 30° | 高斯调谐 |

### STP参数

| 条件 | τ_d | τ_f | 效应 |
|------|-----|-----|------|
| STD主导 | 3.0 s | 0.3 s | 排斥 |
| STF主导 | 0.3 s | 5.0 s | 吸引 |

## 技术特点

- **JAX加速**: 使用`@jax.jit`编译关键函数，支持GPU加速
- **批量处理**: 使用`jax.vmap`实现高效的批量实验
- **FFT卷积**: 使用FFT实现高效的环形卷积
- **模块化设计**: 清晰的模块分离，易于扩展

## 引用

如果您使用本代码，请引用原论文：

```bibtex
@inproceedings{zhang2025serial,
  title={Neural Correlates of Serial Dependence: Synaptic Short-term Plasticity Orchestrates Repulsion and Attraction},
  author={Zhang, Xiuning and Lu, Xincheng and Chen, Nihong and Mi, Yuanyuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 许可证

MIT License

## 联系

如有问题或建议，请提交Issue或Pull Request。


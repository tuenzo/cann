# scripts/run_fig2.py 修复说明

## 问题

运行 `python scripts/run_fig2.py` 时出现错误：
```
TypeError: SingleLayerExperimentConfig.__init__() got an unexpected keyword argument 'n_trials'
```

## 原因

`scripts/run_fig2.py` 使用了旧的 API，而新的 `SingleLayerExperimentConfig` 已经更新：

1. **参数名称改变**：
   - 旧：`n_trials` → 新：`n_runs` 和 `n_trials_per_run`
   - 旧：`isi` → 新：通过 `TrialTimeline` 对象设置

2. **返回数据结构改变**：
   - `run_single_layer_experiment` 现在返回 `trials_df` 和 `curve_binned`
   - `run_experiment_with_recording` 现在返回 `timeseries` 字典

## 修复内容

### 1. 导入更新
```python
from src.experiments.single_layer_exp import (
    SingleLayerExperimentConfig,
    TrialTimeline,  # ← 新增
    run_single_layer_experiment,
    run_experiment_with_recording,
)
```

### 2. 参数解析更新
```python
# 旧
parser.add_argument('--n_trials', type=int, default=20)
config = SingleLayerExperimentConfig(n_trials=args.n_trials, isi=args.isi)

# 新
parser.add_argument('--n_runs', type=int, default=20)
parser.add_argument('--n_trials_per_run', type=int, default=100)
timeline = TrialTimeline(isi=args.isi)
config = SingleLayerExperimentConfig(
    n_runs=args.n_runs,
    n_trials_per_run=args.n_trials_per_run,
    timeline=timeline,
)
```

### 3. 数据访问更新

**实验结果**：
```python
# 旧
std_results['delta']
std_results['errors']

# 新
std_results['trials_df']['delta']
std_results['trials_df']['error']
# 或使用 binned 数据
std_results['curve_binned']['delta']
std_results['curve_binned']['mean_error']
```

**记录数据**：
```python
# 旧
std_recording['time']
std_recording['activity']
std_recording['stp_x']

# 新
std_recording['timeseries']['time']
std_recording['timeseries']['r']
std_recording['timeseries']['stp_x']
```

### 4. 函数调用更新
```python
# 旧
run_experiment_with_recording(config, stp_type='std', delta_to_record=30.0)

# 新
run_experiment_with_recording(
    config=config,
    stp_type='std',
    delta_to_record=-30.0  # 注意：论文要求是 -30°
)
```

## 使用方法

### 完整实验（默认：20 runs × 100 trials）
```bash
python scripts/run_fig2.py --output_dir results/fig2
```

### 快速测试（1 run × 10 trials）
```bash
python scripts/run_fig2.py --n_runs 1 --n_trials_per_run 10 --output_dir results/fig2_test
```

### 自定义 ISI
```bash
python scripts/run_fig2.py --isi 2000.0  # 2秒 ISI
```

## 验证

修复后的脚本已通过：
- ✅ 导入测试
- ✅ 参数解析测试
- ✅ 与新的 API 兼容

## 注意事项

1. **运行时间**：完整实验（20 runs × 100 trials）需要数小时
2. **内存使用**：记录模式会占用较多内存
3. **输出目录**：确保有写入权限


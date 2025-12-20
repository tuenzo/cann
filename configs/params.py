"""
Parameter configurations for the Serial Dependence CANN model.
Based on the parameters from Zhang et al., NeurIPS 2025.
"""

from dataclasses import dataclass, field
from typing import Optional
import jax.numpy as jnp


@dataclass
class NetworkParams:
    """基础网络参数"""
    # 神经元数量和空间范围
    N: int = 180                    # 神经元数量（覆盖180°朝向空间）
    theta_range: float = 180.0      # 朝向空间范围（度）
    
    # 时间参数
    tau: float = 1.0                # 膜电位时间常数 (ms)
    dt: float = 0.1                 # 时间步长 (ms)
    
    # 连接参数
    J0: float = 0.5                 # 递归连接强度
    a: float = 30.0                 # 高斯调谐宽度（度）
    
    # 归一化参数
    k: float = 0.5                  # 除法归一化强度
    rho: float = 1.0                # 神经元密度
    
    # 外部输入
    A: float = 1.0                  # 刺激强度
    stim_width: float = 30.0        # 刺激宽度（度）


@dataclass
class STPParams:
    """短期突触可塑性参数"""
    tau_d: float = 3.0              # STD时间常数 (s)
    tau_f: float = 0.3              # STF时间常数 (s)
    U: float = 0.3                  # 基础释放概率
    
    @classmethod
    def std_dominated(cls):
        """STD主导的参数配置（产生排斥效应）"""
        return cls(tau_d=3.0, tau_f=0.3, U=0.3)
    
    @classmethod
    def stf_dominated(cls):
        """STF主导的参数配置（产生吸引效应）"""
        return cls(tau_d=0.3, tau_f=5.0, U=0.3)


@dataclass
class TwoLayerNetworkParams:
    """两层网络参数"""
    # 下层参数（STD主导 - 感觉皮层）
    low_layer: NetworkParams = field(default_factory=NetworkParams)
    low_stp: STPParams = field(default_factory=STPParams.std_dominated)
    
    # 上层参数（STF主导 - 前额叶）
    high_layer: NetworkParams = field(default_factory=NetworkParams)
    high_stp: STPParams = field(default_factory=STPParams.stf_dominated)
    
    # 层间连接参数
    J_ff: float = 0.3               # 前馈连接强度（下→上）
    J_fb: float = 0.2               # 反馈连接强度（上→下）
    a_ff: float = 30.0              # 前馈连接宽度
    a_fb: float = 30.0              # 反馈连接宽度


@dataclass
class ExperimentParams:
    """实验参数"""
    # 刺激呈现
    stim_duration: float = 500.0    # 刺激持续时间 (ms)
    
    # 时间间隔
    isi: float = 1000.0             # 刺激间隔 ISI (ms)
    iti: float = 3000.0             # 试次间隔 ITI (ms)
    
    # 刺激差异范围
    delta_range: tuple = (-90, 90)  # S1-S2差异范围（度）
    delta_step: float = 5.0         # 差异步长（度）
    
    # 试次数量
    n_trials: int = 20              # 每个条件的试次数
    
    # 随机种子
    seed: int = 42


# 预定义的实验配置
class ExperimentConfigs:
    """预定义实验配置"""
    
    @staticmethod
    def fig2_std():
        """Fig.2 A-C: STD主导单层CANN"""
        return {
            "network": NetworkParams(),
            "stp": STPParams.std_dominated(),
            "experiment": ExperimentParams(isi=1000.0),
        }
    
    @staticmethod
    def fig2_stf():
        """Fig.2 D-F: STF主导单层CANN"""
        return {
            "network": NetworkParams(),
            "stp": STPParams.stf_dominated(),
            "experiment": ExperimentParams(isi=1000.0),
        }
    
    @staticmethod
    def fig3_two_layer():
        """Fig.3: 两层CANN实验"""
        return {
            "network": TwoLayerNetworkParams(),
            "experiment": ExperimentParams(isi=1000.0, iti=3000.0),
        }
    
    @staticmethod
    def fig4_temporal():
        """Fig.4: 时间窗口分析"""
        return {
            "network": TwoLayerNetworkParams(),
            "experiment": ExperimentParams(),
            "isi_range": [500, 1000, 2000, 4000],    # ms
            "iti_range": [1000, 2000, 4000, 8000],   # ms
        }


# 用于单位转换的常量
MS_TO_S = 0.001     # 毫秒转秒
S_TO_MS = 1000.0    # 秒转毫秒
DEG_TO_RAD = jnp.pi / 180.0
RAD_TO_DEG = 180.0 / jnp.pi


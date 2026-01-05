"""
验证 Fig.2 复现代码的验收清单
================================

运行: python verify_checklist.py
"""

import sys
import numpy as np
import jax.numpy as jnp
import jax

# 添加项目路径
sys.path.insert(0, '/home/tuenzh/projects_new/CANN')

from src.experiments.single_layer_exp import (
    SingleLayerExperimentConfig,
    STDConfig,
    STFConfig,
    TrialTimeline,
    run_single_trial,
    run_experiment_with_recording,
    validate_config,
    create_stimulus_with_noise,
    create_noisy_kernel,
    print_validation_report,
)
from src.models.cann import CANNParams, SingleLayerCANN


def run_all_checks():
    """运行所有验收检查"""
    
    print("=" * 70)
    print("Fig.2 复现代码验收清单验证")
    print("=" * 70)
    
    results = {}
    
    # ==========================================================================
    # A. 作用域与运行设置
    # ==========================================================================
    print("\n## A. 作用域与运行设置")
    
    # #1: STD vs STF 选择
    config_std = SingleLayerExperimentConfig(stp_type='std')
    config_stf = SingleLayerExperimentConfig(stp_type='stf')
    net_std = config_std.get_network_config()
    net_stf = config_stf.get_network_config()
    
    check1 = (net_std.tau_d > net_std.tau_f) and (net_stf.tau_f > net_stf.tau_d)
    results['#1 STD/STF选择'] = check1
    print(f"  {'✅' if check1 else '❌'} #1: STD τ_d={net_std.tau_d}s > τ_f={net_std.tau_f}s, "
          f"STF τ_f={net_stf.tau_f}s > τ_d={net_stf.tau_d}s")
    
    # #3: 连接噪声
    check3 = (net_std.mu_J == 0.01)
    results['#3 连接噪声'] = check3
    print(f"  {'✅' if check3 else '❌'} #3: 连接噪声 μ_J = {net_std.mu_J} (应该是 0.01)")
    
    # ==========================================================================
    # B. 网络方程与变量
    # ==========================================================================
    print("\n## B. 网络方程与变量")
    
    # #8: N=100, θ ∈ (-90°, 90°)
    check8a = (net_std.N == 100)
    params = config_std.to_cann_params()
    model = SingleLayerCANN(params)
    check8b = (model.theta[0] == -90.0)
    check8 = check8a and check8b
    results['#8 神经元数和theta范围'] = check8
    print(f"  {'✅' if check8 else '❌'} #8: N={net_std.N} (应该是100), "
          f"θ范围=[{model.theta[0]:.0f}°, {model.theta[-1]:.0f}°)")
    
    # #6: STP变量
    check6 = (hasattr(model.state, 'stp') and 
              hasattr(model.state.stp, 'x') and 
              hasattr(model.state.stp, 'u'))
    results['#6 STP变量'] = check6
    print(f"  {'✅' if check6 else '❌'} #6: STP状态包含 x (shape={model.state.stp.x.shape}) "
          f"和 u (shape={model.state.stp.u.shape})")
    
    # ==========================================================================
    # C. 外部输入与噪声
    # ==========================================================================
    print("\n## C. 外部输入与噪声")
    
    # #10: cue 弱于 stimulus
    check10 = (net_std.alpha_cue < net_std.alpha_sti and
               net_std.a_cue > net_std.a_sti and
               net_std.mu_cue > net_std.mu_sti)
    results['#10 cue弱于stimulus'] = check10
    print(f"  {'✅' if check10 else '❌'} #10: α_cue={net_std.alpha_cue} < α_sti={net_std.alpha_sti}, "
          f"a_cue={net_std.a_cue} > a_sti={net_std.a_sti}")
    
    # #11: 输入参数
    check11 = (net_std.alpha_sti == 20.0 and net_std.a_sti == 0.3 and
               net_std.alpha_cue == 2.5 and net_std.a_cue == 0.4)
    results['#11 输入参数'] = check11
    print(f"  {'✅' if check11 else '❌'} #11: α_sti={net_std.alpha_sti}, a_sti={net_std.a_sti}, "
          f"α_cue={net_std.alpha_cue}, a_cue={net_std.a_cue}")
    
    # ==========================================================================
    # D. 任务时间轴
    # ==========================================================================
    print("\n## D. 任务时间轴")
    
    timeline = TrialTimeline()
    check12 = (timeline.s1_duration == 200.0 and
               timeline.isi == 1000.0 and
               timeline.s2_duration == 200.0 and
               timeline.delay == 3400.0 and
               timeline.cue_duration == 500.0 and
               timeline.iti == 1000.0)
    results['#12 时间轴'] = check12
    print(f"  {'✅' if check12 else '❌'} #12: S1={timeline.s1_duration}ms, ISI={timeline.isi}ms, "
          f"S2={timeline.s2_duration}ms, Delay={timeline.delay}ms, "
          f"Cue={timeline.cue_duration}ms, ITI={timeline.iti}ms")
    print(f"       总时长: {timeline.total_duration()}ms")
    
    # ==========================================================================
    # E. 参数表
    # ==========================================================================
    print("\n## E. 参数表")
    
    # #15: STD参数
    check15 = (net_std.J0 == 0.13 and net_std.a == 0.5 and 
               net_std.k == 0.0018 and net_std.tau_d == 3.0 and
               net_std.tau_f == 0.3 and net_std.U == 0.5)
    results['#15 STD参数'] = check15
    print(f"  {'✅' if check15 else '❌'} #15 STD: J0={net_std.J0}, a={net_std.a}, k={net_std.k}, "
          f"τ_d={net_std.tau_d}s, τ_f={net_std.tau_f}s, U={net_std.U}")
    
    # #16: STF参数
    check16 = (net_stf.J0 == 0.09 and net_stf.a == 0.15 and
               net_stf.k == 0.0095 and net_stf.tau_d == 0.3 and
               net_stf.tau_f == 5.0 and net_stf.U == 0.2)
    results['#16 STF参数'] = check16
    print(f"  {'✅' if check16 else '❌'} #16 STF: J0={net_stf.J0}, a={net_stf.a}, k={net_stf.k}, "
          f"τ_d={net_stf.tau_d}s, τ_f={net_stf.tau_f}s, U={net_stf.U}")
    
    # #17: 时间常数
    check17 = (net_std.tau == 10.0 and net_stf.tau == 10.0)
    results['#17 时间常数'] = check17
    print(f"  {'✅' if check17 else '❌'} #17: τ = {net_std.tau}ms (应该是10ms = 0.01s)")
    
    # ==========================================================================
    # F. 刺激抽样与误差定义
    # ==========================================================================
    print("\n## F. 刺激抽样与误差定义")
    
    config = SingleLayerExperimentConfig()
    check18 = (config.delta_step == 1.0)
    results['#18 步长'] = check18
    print(f"  {'✅' if check18 else '❌'} #18: Δ步长 = {config.delta_step}° (应该是1°)")
    
    # ==========================================================================
    # G. 解码与统计
    # ==========================================================================
    print("\n## G. 解码与统计")
    
    check21 = (config.decode_method == 'pvm')
    results['#21 解码方法'] = check21
    print(f"  {'✅' if check21 else '❌'} #21: 解码方法 = {config.decode_method}")
    
    check22 = (config.n_runs == 20 and config.n_trials_per_run == 100)
    results['#22 运行规模'] = check22
    print(f"  {'✅' if check22 else '❌'} #22: {config.n_runs} runs × {config.n_trials_per_run} trials = "
          f"{config.n_runs * config.n_trials_per_run} 总trials")
    
    # ==========================================================================
    # 功能测试
    # ==========================================================================
    print("\n## 功能测试")
    
    # 测试单次试验运行
    try:
        key = jax.random.PRNGKey(42)
        result = run_single_trial(
            model,
            theta_s1=-30.0,
            theta_s2=0.0,
            config=config_std,
            key=key,
            record=False
        )
        check_trial = ('perceived' in result and 'error' in result and 'delta' in result)
        results['单次试验运行'] = check_trial
        print(f"  {'✅' if check_trial else '❌'} 单次试验: θ_s1=-30°, θ_s2=0°, "
              f"perceived={result['perceived']:.2f}°, error={result['error']:.2f}°")
    except Exception as e:
        results['单次试验运行'] = False
        print(f"  ❌ 单次试验运行失败: {e}")
    
    # 测试带记录的试验
    try:
        result_rec = run_experiment_with_recording(stp_type='std', delta_to_record=-30.0)
        check_rec = ('timeseries' in result_rec and 'cue_activity' in result_rec)
        results['记录模式'] = check_rec
        print(f"  {'✅' if check_rec else '❌'} 记录模式: timeseries有 {len(result_rec['timeseries']['time'])} 时间点")
    except Exception as e:
        results['记录模式'] = False
        print(f"  ❌ 记录模式失败: {e}")
    
    # ==========================================================================
    # 总结
    # ==========================================================================
    print("\n" + "=" * 70)
    print("验收总结")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, passed_check in results.items():
        status = "✅" if passed_check else "❌"
        print(f"  {status} {name}")
    
    print(f"\n通过: {passed}/{total} ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有验收检查通过！")
    else:
        print(f"\n⚠️  有 {total - passed} 项检查未通过，请检查上面的详细信息。")
    
    return results


if __name__ == '__main__':
    run_all_checks()


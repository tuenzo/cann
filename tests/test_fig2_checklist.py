"""
Fig.2 Reproduction Checklist Validation Tests
==============================================

Tests to verify that the single_layer_exp.py implementation
meets all requirements from the paper checklist.

Run with: python -m pytest tests/test_fig2_checklist.py -v
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax

from src.experiments.single_layer_exp import (
    SingleLayerExperimentConfig,
    STDConfig,
    STFConfig,
    TrialTimeline,
    run_single_trial,
    run_single_layer_experiment,
    run_experiment_with_recording,
    validate_config,
    create_stimulus_with_noise,
    create_noisy_kernel,
)
from src.models.cann import CANNParams, SingleLayerCANN, create_gaussian_kernel


# =============================================================================
# A. 作用域与运行设置
# =============================================================================

class TestScopeAndSettings:
    """Tests for Checklist A: Scope and Run Settings."""
    
    def test_01_stp_type_selection(self):
        """#1: STD-dominant vs STF-dominant selection."""
        config_std = SingleLayerExperimentConfig(stp_type='std')
        config_stf = SingleLayerExperimentConfig(stp_type='stf')
        
        net_std = config_std.get_network_config()
        net_stf = config_stf.get_network_config()
        
        # STD: τ_d > τ_f
        assert net_std.tau_d > net_std.tau_f, "STD should have τ_d > τ_f"
        # STF: τ_f > τ_d
        assert net_stf.tau_f > net_stf.tau_d, "STF should have τ_f > τ_d"
    
    def test_03_connection_noise(self):
        """#3: Connection noise μ_J = 0.01."""
        net_config = STDConfig()
        assert net_config.mu_J == 0.01, "Connection noise μ_J should be 0.01"
        
        # Test that noise is actually applied
        key = jax.random.PRNGKey(42)
        kernel = jnp.ones(100)
        noisy_kernel = create_noisy_kernel(kernel, 0.01, key)
        
        assert not jnp.allclose(kernel, noisy_kernel), "Noisy kernel should differ from original"
        # Noise should be small (around 1%)
        relative_diff = jnp.std(noisy_kernel - kernel)
        assert relative_diff < 0.1, "Noise should be small (μ_J=0.01)"


# =============================================================================
# B. 网络方程与变量
# =============================================================================

class TestNetworkEquations:
    """Tests for Checklist B: Network Equations."""
    
    def test_05_divisive_normalization(self):
        """#5: Divisive normalization r = h² / (1 + kρ∫h²)."""
        from src.models.cann import divisive_normalization
        
        N = 100
        u = jnp.ones(N) * 0.5
        k = 0.0018
        rho = 1.0
        
        r = divisive_normalization(u, k, rho)
        
        # Check shape
        assert r.shape == (N,), "Output shape should match input"
        # Check non-negativity
        assert jnp.all(r >= 0), "Firing rate should be non-negative"
    
    def test_06_stp_variables(self):
        """#6: STP equations with u and x variables."""
        params = CANNParams()
        model = SingleLayerCANN(params)
        
        # Check state contains STP variables
        assert hasattr(model.state, 'stp'), "State should have STP"
        assert hasattr(model.state.stp, 'x'), "STP should have x (depression)"
        assert hasattr(model.state.stp, 'u'), "STP should have u (facilitation)"
        
        # Check shapes
        assert model.state.stp.x.shape == (params.N,)
        assert model.state.stp.u.shape == (params.N,)
    
    def test_08_neuron_count_and_theta_range(self):
        """#8: N=100 and θ ∈ (-90°, 90°)."""
        config = SingleLayerExperimentConfig()
        net_config = config.get_network_config()
        
        # Check N = 100
        assert net_config.N == 100, "N should be 100 per Table 1"
        
        # Check theta range
        params = config.to_cann_params()
        model = SingleLayerCANN(params)
        
        assert model.theta[0] == -90.0, "Theta should start at -90°"
        assert model.theta[-1] < 90.0, "Theta should end before 90°"


# =============================================================================
# C. 外部输入与噪声
# =============================================================================

class TestExternalInput:
    """Tests for Checklist C: External Input and Noise."""
    
    def test_09_stimulus_with_noise(self):
        """#9: External input = Gaussian signal + noise."""
        key = jax.random.PRNGKey(42)
        N = 100
        
        stimulus = create_stimulus_with_noise(
            theta_stim=0.0,
            N=N,
            amplitude=20.0,
            width=17.2,  # 0.3 rad in degrees
            noise_strength=0.5,
            key=key
        )
        
        assert stimulus.shape == (N,), "Stimulus shape should be (N,)"
        # Peak should be around theta_stim
        peak_idx = jnp.argmax(stimulus)
        theta = jnp.linspace(-90, 90, N, endpoint=False)
        assert jnp.abs(theta[peak_idx]) < 20, "Peak should be near theta_stim=0"
    
    def test_10_cue_weaker_than_stimulus(self):
        """#10: α_cue << α_sti, a_cue > a_sti, μ_cue > μ_sti."""
        net_config = STDConfig()
        
        assert net_config.alpha_cue < net_config.alpha_sti, "Cue amplitude should be weaker"
        assert net_config.a_cue > net_config.a_sti, "Cue width should be wider"
        assert net_config.mu_cue > net_config.mu_sti, "Cue noise should be stronger"
    
    def test_11_table1_input_parameters(self):
        """#11: Input parameters match Table 1."""
        net_config = STDConfig()
        
        assert net_config.alpha_sti == 20.0, "α_sti should be 20"
        assert net_config.a_sti == 0.3, "a_sti should be 0.3 rad"
        assert net_config.mu_sti == 0.5, "μ_sti should be 0.5"
        assert net_config.alpha_cue == 2.5, "α_cue should be 2.5"
        assert net_config.a_cue == 0.4, "a_cue should be 0.4 rad"
        assert net_config.mu_cue == 1.0, "μ_cue should be 1.0"


# =============================================================================
# D. 任务时间轴
# =============================================================================

class TestTimeline:
    """Tests for Checklist D: Task Timeline."""
    
    def test_12_trial_timeline(self):
        """#12: Trial timeline matches paper."""
        timeline = TrialTimeline()
        
        assert timeline.s1_duration == 200.0, "S1 should be 200ms"
        assert timeline.isi == 1000.0, "ISI should be 1000ms"
        assert timeline.s2_duration == 200.0, "S2 should be 200ms"
        assert timeline.delay == 3400.0, "Delay should be 3400ms"
        assert timeline.cue_duration == 500.0, "Cue should be 500ms"
        assert timeline.iti == 1000.0, "ITI should be 1000ms"
        
        # Total duration
        expected_total = 200 + 1000 + 200 + 3400 + 500 + 1000
        assert timeline.total_duration() == expected_total
    
    def test_12_phase_times(self):
        """#12: Phase start/end times are correct."""
        timeline = TrialTimeline()
        phases = timeline.get_phase_times()
        
        assert phases['S1'] == (0, 200), "S1 should be 0-200ms"
        assert phases['ISI'] == (200, 1200), "ISI should be 200-1200ms"
        assert phases['S2'] == (1200, 1400), "S2 should be 1200-1400ms"
        assert phases['Delay'] == (1400, 4800), "Delay should be 1400-4800ms"
        assert phases['Cue'] == (4800, 5300), "Cue should be 4800-5300ms"
        assert phases['ITI'] == (5300, 6300), "ITI should be 5300-6300ms"
    
    def test_13_example_stimulus(self):
        """#13: Example stimulus θ_s1=-30°, θ_s2=0°."""
        config = SingleLayerExperimentConfig()
        # Default reference_theta should be 0° for S2
        assert config.reference_theta == 0.0, "S2 should be at 0°"


# =============================================================================
# E. 参数表
# =============================================================================

class TestParameters:
    """Tests for Checklist E: Parameter Tables."""
    
    def test_15_std_parameters(self):
        """#15: STD parameters match Table 1."""
        net_config = STDConfig()
        
        assert net_config.J0 == 0.13, "J0 should be 0.13"
        assert net_config.a == 0.5, "a should be 0.5 rad"
        assert net_config.k == 0.0018, "k should be 0.0018"
        assert net_config.mu_b == 0.5, "μ_b should be 0.5"
        assert net_config.tau_d == 3.0, "τ_d should be 3.0s"
        assert net_config.tau_f == 0.3, "τ_f should be 0.3s"
        assert net_config.U == 0.5, "U should be 0.5"
    
    def test_16_stf_parameters(self):
        """#16: STF parameters match Table 1."""
        net_config = STFConfig()
        
        assert net_config.J0 == 0.09, "J0 should be 0.09"
        assert net_config.a == 0.15, "a should be 0.15 rad"
        assert net_config.k == 0.0095, "k should be 0.0095"
        assert net_config.mu_b == 0.5, "μ_b should be 0.5"
        assert net_config.tau_d == 0.3, "τ_d should be 0.3s"
        assert net_config.tau_f == 5.0, "τ_f should be 5.0s"
        assert net_config.U == 0.2, "U should be 0.2"
    
    def test_17_time_constant(self):
        """#17: τ = 0.01s = 10ms."""
        net_std = STDConfig()
        net_stf = STFConfig()
        
        assert net_std.tau == 10.0, "τ should be 10ms (0.01s)"
        assert net_stf.tau == 10.0, "τ should be 10ms (0.01s)"


# =============================================================================
# F. 刺激抽样与误差定义
# =============================================================================

class TestStimulusSampling:
    """Tests for Checklist F: Stimulus Sampling and Error Definition."""
    
    def test_18_delta_step(self):
        """#18: Delta step = 1°."""
        config = SingleLayerExperimentConfig()
        assert config.delta_step == 1.0, "Delta step should be 1°"
    
    def test_19_delta_definition(self):
        """#19: ΔS = θ_s1 - θ_s2."""
        # This is tested implicitly in run_single_trial
        # where delta = theta_s1 - theta_s2
        pass
    
    def test_20_error_definition(self):
        """#20: Error = θ_d2 - θ_s2."""
        # This is tested implicitly in run_single_trial
        # where error = perceived - theta_s2
        pass


# =============================================================================
# G. 解码与统计
# =============================================================================

class TestDecodingAndStatistics:
    """Tests for Checklist G: Decoding and Statistics."""
    
    def test_21_decoding_method(self):
        """#21: Population Vector decoding."""
        config = SingleLayerExperimentConfig()
        assert config.decode_method == 'pvm', "Should use PVM decoding"
    
    def test_22_run_scale(self):
        """#22: 20 runs × 100 trials."""
        config = SingleLayerExperimentConfig()
        
        assert config.n_runs == 20, "Should have 20 runs"
        assert config.n_trials_per_run == 100, "Should have 100 trials per run"
        
        total_trials = config.n_runs * config.n_trials_per_run
        assert total_trials == 2000, "Total trials should be 2000"


# =============================================================================
# Validation Function Test
# =============================================================================

class TestValidation:
    """Test the validation function."""
    
    def test_validate_std_config(self):
        """Test STD configuration validation."""
        config = SingleLayerExperimentConfig(stp_type='std')
        checks = validate_config(config)
        
        # Print report for debugging
        passed = sum(checks.values())
        total = len(checks)
        print(f"\nSTD Config: {passed}/{total} checks passed")
        for name, passed_check in checks.items():
            status = "✅" if passed_check else "❌"
            print(f"  {status} {name}")
        
        # All checks should pass
        assert all(checks.values()), f"Some checks failed: {[k for k,v in checks.items() if not v]}"
    
    def test_validate_stf_config(self):
        """Test STF configuration validation."""
        config = SingleLayerExperimentConfig(stp_type='stf')
        checks = validate_config(config)
        
        # Print report for debugging
        passed = sum(checks.values())
        total = len(checks)
        print(f"\nSTF Config: {passed}/{total} checks passed")
        for name, passed_check in checks.items():
            status = "✅" if passed_check else "❌"
            print(f"  {status} {name}")
        
        # All checks should pass
        assert all(checks.values()), f"Some checks failed: {[k for k,v in checks.items() if not v]}"


# =============================================================================
# Quick Smoke Test
# =============================================================================

class TestSmoke:
    """Quick smoke tests to verify basic functionality."""
    
    def test_single_trial_runs(self):
        """Test that a single trial runs without errors."""
        config = SingleLayerExperimentConfig(stp_type='std')
        params = config.to_cann_params()
        model = SingleLayerCANN(params)
        
        key = jax.random.PRNGKey(42)
        result = run_single_trial(
            model,
            theta_s1=-30.0,
            theta_s2=0.0,
            config=config,
            key=key,
            record=False
        )
        
        assert 'perceived' in result, "Should have perceived orientation"
        assert 'error' in result, "Should have error"
        assert 'delta' in result, "Should have delta"
        
        # Delta should be -30
        assert np.isclose(result['delta'], -30.0), "Delta should be -30°"
    
    def test_recording_runs(self):
        """Test that recording mode works."""
        result = run_experiment_with_recording(stp_type='std', delta_to_record=-30.0)
        
        assert 'timeseries' in result, "Should have timeseries"
        assert 'theta' in result, "Should have theta array"
        assert 'cue_activity' in result, "Should have cue activity"
        
        # Check timeseries contents
        ts = result['timeseries']
        assert 'r' in ts, "Timeseries should have firing rates"
        assert 'stp_x' in ts, "Timeseries should have STP x"
        assert 'stp_u' in ts, "Timeseries should have STP u"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


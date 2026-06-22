"""
Tests für pylabb.control.pid und pylabb.control.stability
"""
import numpy as np
import pytest
from pylabb.control.pid import (
    PIDController,
    DiscretePIDController,
    ziegler_nichols_step,
    ziegler_nichols_oscillation,
    cohen_coon,
    lambda_tuning,
)
from pylabb.control.stability import analyze, bode_margins, critical_gain
from pylabb.core.transfer_function import TransferFunction


class TestPIDController:
    def test_p_controller(self):
        pid = PIDController(Kp=2.0, Ti=0.0, Td=0.0)
        tf = pid.transfer_function()
        assert abs(tf.dc_gain() - 2.0) < 1e-9

    def test_pi_controller_poles(self):
        pid = PIDController(Kp=1.0, Ti=1.0, Td=0.0)
        tf = pid.transfer_function()
        # PI hat einen Pol bei s=0 (Integrator)
        poles = tf.poles()
        assert any(abs(p) < 1e-6 for p in poles)

    def test_repr(self):
        pid = PIDController(Kp=1.5, Ti=2.0, Td=0.1)
        assert "Kp=1.5" in repr(pid)


class TestDiscretePID:
    def test_update_returns_float(self):
        pid = DiscretePIDController(Kp=1.0, Ti=1.0, Td=0.0, dt=0.01)
        u = pid.update(1.0, 0.0)
        assert isinstance(u, float)

    def test_reset_clears_state(self):
        pid = DiscretePIDController(Kp=1.0, Ti=1.0, dt=0.01)
        pid.update(1.0, 0.0)
        pid.reset()
        assert pid._integral == 0.0
        assert pid._prev_error == 0.0

    def test_saturation(self):
        pid = DiscretePIDController(Kp=100.0, u_min=-1.0, u_max=1.0)
        u = pid.update(10.0, 0.0)
        assert u <= 1.0

    def test_simulate_shape(self):
        pid = DiscretePIDController(Kp=1.0, Ti=1.0)
        sp = np.ones(100)
        t_arr, u_arr = pid.simulate(sp)
        assert len(t_arr) == 100
        assert len(u_arr) == 100


class TestTuningRules:
    def test_zn_step_pid(self):
        pid = ziegler_nichols_step(K=1.0, T=1.0, L=0.1, controller_type="PID")
        assert pid.Kp > 0
        assert pid.Ti > 0
        assert pid.Td > 0

    def test_zn_oscillation(self):
        pid = ziegler_nichols_oscillation(Ku=10.0, Tu=2.0, controller_type="PID")
        assert pid.Kp == pytest.approx(6.0)

    def test_cohen_coon(self):
        pid = cohen_coon(K=1.0, T=1.0, L=0.1, controller_type="PI")
        assert pid.Kp > 0

    def test_lambda_tuning(self):
        pid = lambda_tuning(K=1.0, T=1.0, L=0.1)
        assert pid.Ti > 0


class TestStabilityAnalysis:
    def _pt2(self):
        return TransferFunction([1], [1, 1.4, 1])

    def test_is_stable(self):
        G = self._pt2()
        info = analyze(G)
        assert info.is_stable

    def test_rhp_poles_zero(self):
        info = analyze(self._pt2())
        assert info.rhp_poles == 0

    def test_unstable_system(self):
        G = TransferFunction([1], [1, -1, 0])  # Pole bei 0, +1
        info = analyze(G)
        assert not info.is_stable
        assert info.rhp_poles >= 1

    def test_margins_finite(self):
        # PT3: ausreichend Phase für endliche Ränder
        G = TransferFunction([1], [1, 3, 3, 1])  # (s+1)³
        info = analyze(G)
        # Endliche Ränder erwartet
        assert np.isfinite(info.gain_margin_dB) or np.isfinite(info.phase_margin_deg)

    def test_bode_margins_stable(self):
        G = TransferFunction([1], [1, 2, 1])
        gm, pm, wgc, wpc = bode_margins(G)
        # Stabiles System hat positiven Phasenrand
        # (kann inf sein, wenn kein 0-dB-Durchgang)
        assert pm > 0 or pm == float("inf")

    def test_critical_gain(self):
        G = TransferFunction([1], [1, 3, 3, 1])
        Ku, wu = critical_gain(G)
        assert Ku > 0
        assert wu > 0

"""
Erweiterte Tests für pylabb.control.pid – 100 % Pfadabdeckung
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


class TestPIDControllerTransferFunction:
    def test_p_controller(self):
        pid = PIDController(Kp=3.0, Ti=0.0, Td=0.0)
        tf = pid.transfer_function()
        assert abs(tf.dc_gain() - 3.0) < 1e-9

    def test_pi_controller(self):
        pid = PIDController(Kp=1.0, Ti=2.0, Td=0.0)
        tf = pid.transfer_function()
        # PI has integrator: pole at s=0
        poles = tf.poles()
        assert any(abs(p) < 1e-6 for p in poles)

    def test_pd_controller(self):
        """Ti=0, Td>0 → PD path."""
        pid = PIDController(Kp=1.0, Ti=0.0, Td=0.5, N=10.0)
        tf = pid.transfer_function()
        # PD: has gain at DC
        assert abs(tf.dc_gain()) > 0

    def test_full_pid_controller(self):
        pid = PIDController(Kp=1.0, Ti=2.0, Td=0.5, N=10.0)
        tf = pid.transfer_function()
        poles = tf.poles()
        assert any(abs(p) < 1e-6 for p in poles)

    def test_repr(self):
        pid = PIDController(Kp=2.5, Ti=1.0, Td=0.1, N=15.0, name="MyPID")
        r = repr(pid)
        assert "MyPID" in r
        assert "Kp=2.5" in r


class TestDiscretePIDExtended:
    def test_with_td_nonzero(self):
        """Cover D-filter branch (Td > 0)."""
        pid = DiscretePIDController(Kp=1.0, Ti=2.0, Td=0.1, dt=0.01)
        u1 = pid.update(1.0, 0.0)
        u2 = pid.update(1.0, 0.5)
        # Second update has smaller error → smaller u expected
        assert u2 < u1

    def test_no_antiwindup(self):
        pid = DiscretePIDController(Kp=1.0, Ti=1.0, dt=0.01, anti_windup=False)
        u = pid.update(1.0, 0.0)
        assert isinstance(u, float)

    def test_ti_infinity(self):
        """Ti=inf → no integral action."""
        pid = DiscretePIDController(Kp=2.0, Ti=float("inf"), dt=0.01)
        u = pid.update(1.0, 0.0)
        assert abs(u - 2.0) < 1e-9  # only P action

    def test_simulate_with_explicit_t(self):
        pid = DiscretePIDController(Kp=1.0, Ti=1.0, dt=0.01)
        n = 50
        t = np.arange(n) * 0.01
        sp = np.ones(n)
        t_out, u_out = pid.simulate(sp, t=t)
        assert len(t_out) == n
        assert np.allclose(t_out, t)

    def test_to_params(self):
        pid = DiscretePIDController(Kp=1.5, Ti=2.0, Td=0.1, dt=0.01,
                                    u_min=-5.0, u_max=5.0, N=15.0)
        params = pid.to_params()
        assert params["Kp"] == 1.5
        assert params["Ti"] == 2.0
        assert params["Td"] == 0.1
        assert params["dt"] == 0.01
        assert params["u_min"] == -5.0
        assert params["u_max"] == 5.0
        assert params["N"] == 15.0

    def test_repr(self):
        pid = DiscretePIDController(Kp=1.0, Ti=1.0, Td=0.1, dt=0.01)
        r = repr(pid)
        assert "DiscretePIDController" in r

    def test_lower_saturation(self):
        pid = DiscretePIDController(Kp=100.0, Ti=float("inf"),
                                    u_min=-1.0, u_max=1.0)
        u = pid.update(-10.0, 0.0)  # large negative error
        assert u >= -1.0


class TestZieglerNicholsTypes:
    def test_zn_step_P(self):
        pid = ziegler_nichols_step(K=1.0, T=1.0, L=0.1, controller_type="P")
        assert pid.Ti == 0.0
        assert pid.Td == 0.0

    def test_zn_step_PI(self):
        pid = ziegler_nichols_step(K=1.0, T=1.0, L=0.1, controller_type="PI")
        assert pid.Ti > 0
        assert pid.Td == 0.0

    def test_zn_oscillation_P(self):
        pid = ziegler_nichols_oscillation(Ku=10.0, Tu=2.0, controller_type="P")
        assert abs(pid.Kp - 5.0) < 0.01
        assert pid.Ti == 0.0

    def test_zn_oscillation_PI(self):
        pid = ziegler_nichols_oscillation(Ku=10.0, Tu=2.0, controller_type="PI")
        assert pid.Ti > 0
        assert pid.Td == 0.0


class TestCohenCoonTypes:
    def test_cc_P(self):
        pid = cohen_coon(K=1.0, T=1.0, L=0.1, controller_type="P")
        assert pid.Ti == 0.0
        assert pid.Td == 0.0

    def test_cc_PI(self):
        pid = cohen_coon(K=1.0, T=1.0, L=0.1, controller_type="PI")
        assert pid.Ti > 0

    def test_cc_PD(self):
        pid = cohen_coon(K=1.0, T=1.0, L=0.1, controller_type="PD")
        assert pid.Td > 0
        assert pid.Ti == 0.0

    def test_cc_PID(self):
        pid = cohen_coon(K=1.0, T=1.0, L=0.1, controller_type="PID")
        assert pid.Kp > 0
        assert pid.Ti > 0
        assert pid.Td > 0


class TestLambdaTuning:
    def test_default_lambda(self):
        pid = lambda_tuning(K=1.0, T=2.0, L=0.1)
        assert pid.Ti > 0

    def test_custom_lambda(self):
        pid = lambda_tuning(K=1.0, T=2.0, L=0.1, lambda_=1.0)
        assert pid.Ti > 0

    def test_result_is_pi(self):
        pid = lambda_tuning(K=1.0, T=1.0, L=0.2)
        assert pid.Td == 0.0  # λ-tuning yields PI

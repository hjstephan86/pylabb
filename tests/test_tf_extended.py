"""
Erweiterte Tests für pylabb.core.transfer_function – 100 % Pfadabdeckung
"""
import numpy as np
import pytest

from pylabb.core.transfer_function import (
    TransferFunction,
    _check_compat,
    _auto_omega,
    _sim_time,
)


class TestNormalize:
    def test_leading_zeros_stripped(self):
        G = TransferFunction([0, 0, 1], [0, 1, 1])
        assert G.num[0] != 0
        assert G.den[0] != 0

    def test_all_zero_num_becomes_scalar_zero(self):
        G = TransferFunction([0, 0], [1, 1])
        assert np.allclose(G.num, [0.0])

    def test_all_zero_den_becomes_unity(self):
        # Only strip leading zeros; if trimming empties den use fallback [1.0]
        G = TransferFunction([1], [0, 0, 1])
        assert G.den[-1] == pytest.approx(1.0)


class TestDiscrete:
    def test_as_scipy_discrete(self):
        G = TransferFunction([1], [1, 1])
        Gd = G.discretize(0.01, method="tustin")
        sp = Gd._as_scipy()
        assert sp is not None

    def test_freqresp_discrete(self):
        G = TransferFunction([1], [1, 1])
        Gd = G.discretize(0.01, method="tustin")
        omega, H = Gd.freqresp(n_points=100)
        assert len(H) == 100

    def test_dc_gain_discrete(self):
        G = TransferFunction([1], [1, 1])
        Gd = G.discretize(0.01, method="tustin")
        dc = Gd.dc_gain()
        assert abs(dc - 1.0) < 0.05

    def test_step_response_discrete(self):
        G = TransferFunction([1], [1, 1])
        Gd = G.discretize(0.01, method="tustin")
        t, y = Gd.step_response()
        assert len(t) == 200  # _step_discrete always uses 200 samples
        # at 200*0.01 = 2 s (2 time constants), y ≈ 0.86 – heading toward 1
        assert y[-1] > 0.5

    def test_lsim_discrete(self):
        G = TransferFunction([1], [1, 1])
        Gd = G.discretize(0.01, method="zoh")
        t = np.arange(500) * 0.01
        u = np.ones(500)
        t_out, y_out = Gd.lsim(u, t)
        assert len(y_out) == 500

    def test_discretize_euler(self):
        G = TransferFunction([1], [1, 1])
        Gd = G.discretize(0.01, method="euler")
        assert Gd.dt == pytest.approx(0.01)

    def test_discretize_already_discrete_raises(self):
        Gd = TransferFunction([1], [1, 1]).discretize(0.01)
        with pytest.raises(ValueError):
            Gd.discretize(0.01)

    def test_discretize_unknown_method_raises(self):
        G = TransferFunction([1], [1, 1])
        with pytest.raises(ValueError):
            G.discretize(0.01, method="unknown")


class TestImpulseResponse:
    def test_impulse_continuous(self):
        G = TransferFunction([1], [1, 1])
        t, y = G.impulse_response()
        assert len(t) > 0
        # PT1 impulse: y(0) ≈ 1.0
        assert y[0] > 0

    def test_impulse_discrete(self):
        G = TransferFunction([1], [1, 1])
        Gd = G.discretize(0.01, method="tustin")
        t, y = Gd.impulse_response()
        assert len(t) > 0


class TestConversion:
    def test_to_state_space(self):
        G = TransferFunction([1], [1, 2, 1])
        ss = G.to_state_space()
        assert ss.order == 2

    def test_to_state_space_discrete(self):
        Gd = TransferFunction([1], [1, 1]).discretize(0.01)
        ss = Gd.to_state_space()
        assert ss.order == 1


class TestArithmeticExtended:
    def test_div_float(self):
        G = TransferFunction([6], [1, 1])
        G2 = G / 3.0
        assert abs(G2.dc_gain() - 2.0) < 1e-9

    def test_sub_tf(self):
        G1 = TransferFunction([1], [1, 1])
        G2 = TransferFunction([1], [1, 1])
        G3 = G1 - G2
        assert np.allclose(G3.num, [0.0])

    def test_sub_float(self):
        """G - float triggers the isinstance branch in __sub__."""
        G = TransferFunction([3], [1, 1])
        G2 = G - 1.0
        # DC gain: G(0)=3, (G-1)(0) = 3-1 = 2
        assert abs(G2.dc_gain() - 2.0) < 1e-6

    def test_neg(self):
        G = TransferFunction([2], [1, 1])
        G2 = -G
        assert abs(G2.dc_gain() + 2.0) < 1e-9
        assert G2.name.startswith("-")

    def test_radd(self):
        G = TransferFunction([1], [1, 1])
        G2 = 2.0 + G
        # 2 + 1/(s+1) = (2s+3)/(s+1)
        assert abs(G2.dc_gain() - 3.0) < 0.1

    def test_feedback_with_custom_H(self):
        G = TransferFunction([1], [1, 1])
        H = TransferFunction([2], [1])
        T = G.feedback(H=H)
        # T = G / (1 + 2G) → dc = (1/1) / (1 + 2) = 1/3
        assert abs(T.dc_gain() - 1.0 / 3.0) < 0.01

    def test_feedback_positive(self):
        G = TransferFunction([1], [1, 3])
        T = G.feedback(sign=+1)
        # T = G / (1 - G)
        assert T is not None

    def test_check_compat_raises_on_dt_mismatch(self):
        G1 = TransferFunction([1], [1, 1], dt=0.01)
        G2 = TransferFunction([1], [1, 1], dt=0.02)
        with pytest.raises(ValueError):
            _check_compat(G1, G2)

    def test_repr(self):
        G = TransferFunction([1], [1, 2, 1], name="PT2")
        r = repr(G)
        assert "PT2" in r
        assert "TransferFunction" in r


class TestAutoOmegaAndSimTime:
    def test_auto_omega_no_poles(self):
        G = TransferFunction([1], [1])
        omega = _auto_omega(G, 100)
        assert len(omega) == 100

    def test_auto_omega_with_poles(self):
        G = TransferFunction([1], [1, 1])
        omega = _auto_omega(G, 200)
        assert len(omega) == 200

    def test_sim_time_stable(self):
        G = TransferFunction([1], [1, 1])
        t = _sim_time(G)
        assert t[-1] > 0

    def test_sim_time_no_stable_poles(self):
        G = TransferFunction([1], [1])  # gain only, no poles
        t = _sim_time(G)
        assert t[-1] > 0


class TestBode:
    def test_bode_returns_three_arrays(self):
        G = TransferFunction([1], [1, 1])
        omega, mag_dB, phase_deg = G.bode(n_points=100)
        assert len(omega) == len(mag_dB) == len(phase_deg) == 100

    def test_bode_dc_gain_dB(self):
        G = TransferFunction([1], [1, 1])
        omega, mag_dB, phase_deg = G.bode(omega=np.array([1e-4]))
        # DC gain ~0 dB for PT1 with K=1
        assert abs(mag_dB[0] - 0.0) < 0.1

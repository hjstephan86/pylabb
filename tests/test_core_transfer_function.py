"""
Tests für pylabb.core.transfer_function
"""
import numpy as np
import pytest
from pylabb.core.transfer_function import TransferFunction


class TestTransferFunctionBasics:
    def test_creation(self):
        G = TransferFunction([1], [1, 1])
        assert G.name == "G"
        assert len(G.poles()) == 1

    def test_pt1_gain(self):
        # G = K / (Ts + 1) → dc_gain = K
        G = TransferFunction([3], [2, 1])
        assert abs(G.dc_gain() - 3.0) < 1e-9

    def test_poles(self):
        G = TransferFunction([1], [1, 3, 2])  # (s+1)(s+2)
        poles = sorted(G.poles().real)
        assert poles[0] == pytest.approx(-2.0, abs=1e-6)
        assert poles[1] == pytest.approx(-1.0, abs=1e-6)

    def test_zeros(self):
        G = TransferFunction([1, 1], [1, 2, 1])  # (s+1)/(s+1)² → Nullstelle bei -1
        zeros = G.zeros()
        assert len(zeros) == 1
        assert zeros[0].real == pytest.approx(-1.0, abs=1e-6)

    def test_is_stable_true(self):
        G = TransferFunction([1], [1, 1])
        assert G.is_stable()

    def test_is_stable_false(self):
        G = TransferFunction([1], [1, -1])  # Pol bei +1
        assert not G.is_stable()


class TestTransferFunctionArithmetic:
    def test_series(self):
        G1 = TransferFunction([1], [1, 1])
        G2 = TransferFunction([1], [1, 2])
        G = G1 * G2
        assert len(G.poles()) == 2

    def test_parallel(self):
        G1 = TransferFunction([1], [1, 1])
        G2 = TransferFunction([1], [1, 1])
        G = G1 + G2
        # 2 / (s+1)
        assert abs(G.dc_gain() - 2.0) < 1e-9

    def test_feedback(self):
        G = TransferFunction([1], [1, 1, 0])  # 1/(s²+s)
        T = G.feedback()  # T = G/(1+G)
        # Pole des geschlossenen Kreises
        assert T.is_stable()

    def test_scalar_multiply(self):
        G = TransferFunction([1], [1, 1])
        G2 = 3.0 * G
        assert abs(G2.dc_gain() - 3.0) < 1e-9


class TestTransferFunctionFreqResp:
    def test_freqresp_shape(self):
        G = TransferFunction([1], [1, 1])
        omega, H = G.freqresp(n_points=100)
        assert len(omega) == 100
        assert len(H) == 100

    def test_bode_dc_gain(self):
        # Bei ω→0 soll der Betrag dem DC-Gain entsprechen
        G = TransferFunction([2], [1, 1])
        omega, mag_dB, phase_deg = G.bode(n_points=500)
        # Bei sehr kleinen Frequenzen: mag ≈ 20*log10(2) ≈ 6.02 dB
        assert abs(mag_dB[0] - 20 * np.log10(2)) < 0.5


class TestTransferFunctionTimeResponse:
    def test_step_response_shape(self):
        G = TransferFunction([1], [1, 1])
        t, y = G.step_response()
        assert len(t) == len(y)
        assert len(t) > 10

    def test_step_response_final_value(self):
        G = TransferFunction([2], [1, 2])  # ω=2, dc=1
        t, y = G.step_response(t_end=20)
        assert abs(y[-1] - 1.0) < 0.05  # Grenzwert = 1

    def test_lsim(self):
        G = TransferFunction([1], [1, 1])
        t = np.linspace(0, 5, 500)
        u = np.ones_like(t)
        t_out, y_out = G.lsim(u, t)
        assert len(t_out) == len(y_out)


class TestDiscretize:
    def test_discretize_tustin(self):
        G = TransferFunction([1], [1, 1])
        G_d = G.discretize(0.01, method="tustin")
        assert G_d.dt == pytest.approx(0.01)
        assert G_d.is_stable()

    def test_discretize_zoh(self):
        G = TransferFunction([1], [1, 1])
        G_d = G.discretize(0.01, method="zoh")
        assert G_d.dt == pytest.approx(0.01)

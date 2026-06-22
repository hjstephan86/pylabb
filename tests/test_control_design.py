"""
Tests für pylabb.control.design – 100 % Pfadabdeckung
"""
import numpy as np
import pytest

from pylabb.control.design import (
    lead_compensator,
    lag_compensator,
    lead_lag_compensator,
    notch_filter,
    bandpass_filter,
    lowpass_filter,
    highpass_filter,
    closed_loop,
    sensitivity,
    complementary_sensitivity,
    control_sensitivity,
    gain_to_dB,
    dB_to_gain,
    series,
    parallel,
    pade_approximation,
    first_order_plant,
    second_order_plant,
    integrating_plant,
)
from pylabb.core.transfer_function import TransferFunction


class TestLeadLagCompensator:
    def test_lead_name(self):
        C = lead_compensator(alpha=5.0, T=0.1)
        assert C.name == "Lead"

    def test_lag_name(self):
        C = lead_compensator(alpha=0.2, T=1.0)
        assert C.name == "Lag"

    def test_lag_compensator_wrapper(self):
        C = lag_compensator(alpha=0.5, T=1.0, K=2.0)
        dc = C.dc_gain()
        assert abs(dc - 2.0) < 0.01

    def test_lag_compensator_alpha_ge_1_raises(self):
        with pytest.raises(AssertionError):
            lag_compensator(alpha=2.0, T=1.0)

    def test_lead_lag_combination(self):
        C = lead_lag_compensator(T1=0.1, alpha1=5.0, T2=1.0, alpha2=0.2)
        # Resultant is a product; just check it's a valid TF
        dc = C.dc_gain()
        assert np.isfinite(abs(dc))

    def test_gain_scaling(self):
        C = lead_compensator(alpha=2.0, T=1.0, K=3.0)
        # DC gain = K * 1 / alpha = 3/2 (pole and zero don't cancel)
        dc = abs(C.dc_gain())
        assert dc > 0


class TestFilters:
    def test_notch_filter_notches_at_omega_n(self):
        omega_n = 10.0
        C = notch_filter(omega_n, zeta_z=0.01, zeta_p=0.3)
        # At omega_n the gain should be heavily attenuated
        _, H = C.freqresp(omega=np.array([omega_n]))
        assert abs(H[0]) < 0.2

    def test_bandpass_peak_near_omega_c(self):
        omega_c = 5.0
        C = bandpass_filter(omega_c, Q=10.0)
        omega_test = np.logspace(-1, 2, 500)
        _, H = C.freqresp(omega=omega_test)
        peak_idx = np.argmax(np.abs(H))
        assert abs(omega_test[peak_idx] - omega_c) / omega_c < 0.2

    def test_lowpass_order1_dc_gain(self):
        C = lowpass_filter(omega_c=100.0, order=1)
        assert abs(C.dc_gain() - 1.0) < 0.01

    def test_lowpass_order2_dc_gain(self):
        C = lowpass_filter(omega_c=100.0, order=2)
        assert abs(C.dc_gain() - 1.0) < 0.01

    def test_lowpass_invalid_order(self):
        with pytest.raises(ValueError):
            lowpass_filter(omega_c=10.0, order=3)

    def test_highpass_attenuates_dc(self):
        C = highpass_filter(omega_c=1.0)
        # DC gain should be ~0
        _, H = C.freqresp(omega=np.array([1e-4]))
        assert abs(H[0]) < 0.01


class TestSensitivityFunctions:
    def _plant_controller(self):
        G = TransferFunction([1], [1, 2, 1])
        C = TransferFunction([2], [1])
        return G, C

    def test_closed_loop_dc_finite(self):
        """Closed-loop DC gain is C(0)*G(0)/(1+C(0)*G(0)) = 2/3 for this plant."""
        G, C = self._plant_controller()
        T = closed_loop(G, C)
        dc = abs(T.dc_gain())
        expected = 2.0 / 3.0
        assert abs(dc - expected) < 0.05

    def test_closed_loop_with_H(self):
        G, C = self._plant_controller()
        H = TransferFunction([2], [1])
        T = closed_loop(G, C, H=H)
        assert T is not None

    def test_sensitivity_dc_near_zero(self):
        G, C = self._plant_controller()
        S = sensitivity(G, C)
        dc = abs(S.dc_gain())
        assert dc < 0.5  # loop gain > 1 → S(0) small

    def test_complementary_sensitivity(self):
        G, C = self._plant_controller()
        T = complementary_sensitivity(G, C)
        S = sensitivity(G, C)
        # S + T = 1 at every frequency
        _, Hv_s = S.freqresp(n_points=100)
        _, Hv_t = T.freqresp(n_points=100)
        assert np.allclose(np.abs(Hv_s + Hv_t), 1.0, atol=1e-6)

    def test_control_sensitivity(self):
        G, C = self._plant_controller()
        Q = control_sensitivity(G, C)
        assert Q is not None


class TestGainConversions:
    def test_gain_to_dB(self):
        assert abs(gain_to_dB(10.0) - 20.0) < 1e-9

    def test_dB_to_gain(self):
        assert abs(dB_to_gain(20.0) - 10.0) < 1e-9

    def test_roundtrip(self):
        g = 3.7
        assert abs(dB_to_gain(gain_to_dB(g)) - g) < 1e-9


class TestSeriesParallel:
    def test_series_two(self):
        G1 = TransferFunction([1], [1, 1])
        G2 = TransferFunction([2], [1, 2])
        G = series(G1, G2)
        # DC gain = 1 * 1 = 1
        assert abs(G.dc_gain() - 1.0) < 0.01

    def test_series_three(self):
        G1 = TransferFunction([2], [1])
        G2 = TransferFunction([3], [1])
        G3 = TransferFunction([1], [1])
        G = series(G1, G2, G3)
        assert abs(G.dc_gain() - 6.0) < 0.01

    def test_parallel_two(self):
        G1 = TransferFunction([1], [1])  # gain 1
        G2 = TransferFunction([2], [1])  # gain 2
        G = parallel(G1, G2)
        assert abs(G.dc_gain() - 3.0) < 0.01


class TestPadeApproximation:
    def test_order_1(self):
        G = pade_approximation(L=0.1, order=1)
        # |G(0)| = 1 and it's a real rational function
        assert abs(abs(G.dc_gain()) - 1.0) < 0.01

    def test_order_2(self):
        G = pade_approximation(L=0.1, order=2)
        assert abs(abs(G.dc_gain()) - 1.0) < 0.01

    def test_order_3(self):
        G = pade_approximation(L=0.1, order=3)
        assert G is not None

    def test_order_4(self):
        G = pade_approximation(L=0.1, order=4)
        assert G is not None

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            pade_approximation(L=0.1, order=5)


class TestStandardPlants:
    def test_fopdt_no_delay(self):
        G = first_order_plant(K=2.0, T=1.0, L=0.0)
        assert abs(G.dc_gain() - 2.0) < 0.01

    def test_fopdt_with_delay(self):
        G = first_order_plant(K=1.0, T=1.0, L=0.2)
        # DC gain should still be ~1 (Padé at s=0 ≈ 1)
        assert abs(abs(G.dc_gain()) - 1.0) < 0.05

    def test_second_order_no_delay(self):
        G = second_order_plant(K=1.0, omega_n=2.0, zeta=0.7, L=0.0)
        assert abs(G.dc_gain() - 1.0) < 0.01

    def test_second_order_with_delay(self):
        G = second_order_plant(K=1.0, omega_n=2.0, zeta=0.7, L=0.1)
        assert G is not None

    def test_integrating_plant_no_T(self):
        G = integrating_plant(K=1.0, T=0.0)
        # G = K/s → pole at 0
        poles = G.poles()
        assert any(abs(p) < 1e-6 for p in poles)

    def test_integrating_plant_with_T(self):
        G = integrating_plant(K=1.0, T=1.0)
        # G = K / (s(Ts+1)) → poles at 0 and -1/T
        poles = G.poles()
        assert any(abs(p) < 1e-6 for p in poles)

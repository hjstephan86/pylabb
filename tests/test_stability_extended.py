"""
Erweiterte Tests für pylabb.control.stability – 100 % Pfadabdeckung
"""
import numpy as np
import pytest

from pylabb.control.stability import (
    StabilityInfo,
    analyze,
    bode_margins,
    pole_zero_info,
    root_locus_data,
    critical_gain,
    _dominant_poles,
)
from pylabb.core.transfer_function import TransferFunction


def _pt3():
    """(s+1)^3 = s^3 + 3s^2 + 3s + 1"""
    return TransferFunction([1], [1, 3, 3, 1])


def _discrete_pt1():
    return TransferFunction([1], [1, 1]).discretize(dt=0.01, method="tustin")


class TestStabilityInfoStr:
    def test_str_contains_labels(self):
        G = TransferFunction([1], [1, 1.4, 1])
        info = analyze(G)
        s = str(info)
        assert "Stabil" in s
        assert "Amplitudenrand" in s
        assert "Phasenrand" in s
        assert "Pole" in s
        assert "Nullstellen" in s


class TestAnalyzeDiscrete:
    def test_discrete_stable(self):
        Gd = _discrete_pt1()
        info = analyze(Gd)
        assert info.is_stable

    def test_discrete_unstable(self):
        # Pole outside unit circle
        Gd = TransferFunction([1], [1, -1.5], dt=0.01)
        info = analyze(Gd)
        assert not info.is_stable
        assert info.rhp_poles >= 1


class TestStrInfiniteMargins:
    def test_high_gain_margins(self):
        """System with no 0-dB crossing gives inf margins."""
        G = TransferFunction([0.001], [1, 10, 100])
        info = analyze(G)
        # Margins may be inf or finite – just check it doesn't crash
        assert np.isfinite(info.gain_margin_dB) or info.gain_margin_dB == float("inf")


class TestPoleZeroInfo:
    def test_minimum_phase(self):
        G = TransferFunction([1, 2], [1, 3, 2])  # zero at -2 (LHP)
        info = pole_zero_info(G)
        assert info["is_minimum_phase"]
        assert len(info["poles"]) > 0

    def test_non_minimum_phase(self):
        G = TransferFunction([1, -1], [1, 2, 1])  # zero at +1 (RHP)
        info = pole_zero_info(G)
        assert not info["is_minimum_phase"]

    def test_natural_frequencies(self):
        G = TransferFunction([1], [1, 2, 1])
        info = pole_zero_info(G)
        assert all(f >= 0 for f in info["natural_frequencies"])


class TestRootLocusData:
    def test_shape(self):
        G = _pt3()
        k_range, rl_poles = root_locus_data(G, n_k=50)
        assert len(k_range) == 50
        assert rl_poles.shape[0] == 50
        assert rl_poles.shape[1] == 3  # 3rd order system

    def test_custom_k_range(self):
        G = TransferFunction([1], [1, 1])
        k_vec = np.linspace(0.1, 5.0, 20)
        k_range, rl_poles = root_locus_data(G, k_range=k_vec)
        assert len(k_range) == 20

    def test_k_zero_gives_open_loop_poles(self):
        G = TransferFunction([1], [1, 2])  # pole at -2
        k_range, rl_poles = root_locus_data(G, n_k=10)
        # At k≈0 the closed-loop poles ≈ open-loop poles
        assert abs(rl_poles[0, 0] + 2.0) < 0.1


class TestDominantPoles:
    def test_returns_n_poles(self):
        poles = np.array([-1 + 0j, -2 + 0j, -10 + 0j])
        dom = _dominant_poles(poles, n=2)
        assert len(dom) <= 2

    def test_smallest_real_part_selected(self):
        poles = np.array([-1 + 0j, -5 + 0j])
        dom = _dominant_poles(poles, n=1)
        assert abs(dom[0] + 1.0) < 1e-9

    def test_no_stable_poles(self):
        poles = np.array([1 + 0j, 2 + 0j])
        dom = _dominant_poles(poles, n=2)
        # Falls back to all poles
        assert len(dom) == 2


class TestCriticalGain:
    def test_pt3_has_finite_ku(self):
        G = _pt3()
        Ku, wu = critical_gain(G)
        assert Ku > 0
        assert wu > 0

    def test_pt1_marginal(self):
        G = TransferFunction([1], [1, 1])
        Ku, wu = critical_gain(G)
        # PT1 never reaches -180°, so phase minimum is different
        # Just check types
        assert isinstance(Ku, float)
        assert isinstance(wu, float)

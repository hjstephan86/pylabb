"""
Tests für pylabb.core.math_utils – 100 % Pfadabdeckung
"""
import numpy as np
import pytest

from pylabb.core.math_utils import (
    poly_eval,
    poly_multiply,
    poly_add,
    roots_of_polynomial,
    partial_fraction,
    bilinear_transform,
    zoh_discretize,
    routh_array,
    count_rhp_roots,
)


class TestPolyOps:
    def test_poly_eval_scalar(self):
        # x^2 - 3x + 2 = 0 at x=1 and x=2
        assert abs(poly_eval([1, -3, 2], 1.0)) < 1e-12
        assert abs(poly_eval([1, -3, 2], 2.0)) < 1e-12

    def test_poly_eval_array(self):
        x = np.array([0.0, 1.0, 2.0])
        result = poly_eval([1, 0, 0], x)  # x^2
        assert np.allclose(result, x**2)

    def test_poly_multiply(self):
        # (s+1)*(s+2) = s^2 + 3s + 2
        result = poly_multiply([1, 1], [1, 2])
        assert np.allclose(result, [1, 3, 2])

    def test_poly_add(self):
        # (s^2 + 1) + (s + 1) = s^2 + s + 2
        result = poly_add([1, 0, 1], [1, 1])
        assert np.allclose(result, [1, 1, 2])

    def test_roots_of_polynomial(self):
        # s^2 - 1 = 0 → roots ±1
        r = roots_of_polynomial([1, 0, -1])
        assert set(np.round(r.real, 8)) == {1.0, -1.0}


class TestPartialFraction:
    def test_simple_poles(self):
        # 1 / (s+1)(s+2) → residues and poles
        residues, poles, k = partial_fraction([1], [1, 3, 2])
        assert len(poles) == 2
        # Poles at -1 and -2
        assert any(abs(p + 1) < 1e-6 for p in poles)
        assert any(abs(p + 2) < 1e-6 for p in poles)


class TestBilinearTransform:
    def test_pt1_result_length(self):
        # H(s) = 1/(s+1) discretized
        num_d, den_d = bilinear_transform([1], [1, 1], fs=100.0)
        assert len(num_d) == 2
        assert len(den_d) == 2

    def test_pt1_dc_gain_preserved(self):
        # DC gain H(z=1) = sum(b) / sum(a) = 1 / (1+1) * 2 / ... ≈ 1
        num_d, den_d = bilinear_transform([1], [1, 1], fs=10.0)
        dc = sum(num_d) / sum(den_d)
        assert abs(dc - 1.0) < 0.01


class TestZOHDiscretize:
    def test_shape(self):
        A = np.array([[0.0, 1.0], [-1.0, -1.0]])
        B = np.array([[0.0], [1.0]])
        Ad, Bd = zoh_discretize(A, B, dt=0.01)
        assert Ad.shape == (2, 2)
        assert Bd.shape == (2, 1)

    def test_ad_near_identity_small_dt(self):
        A = np.array([[-1.0]])
        B = np.array([[1.0]])
        Ad, Bd = zoh_discretize(A, B, dt=0.001)
        # e^{-0.001} ≈ 0.999
        assert abs(Ad[0, 0] - np.exp(-0.001)) < 1e-6


class TestRouthArray:
    def test_stable_2nd_order(self):
        # s^2 + 2s + 1 → all rows positive → stable
        table = routh_array([1, 2, 1])
        assert table.shape[0] == 3

    def test_stable_3rd_order(self):
        # s^3 + 2s^2 + 3s + 4 (has RHP roots, but array is computable)
        table = routh_array([1, 2, 3, 4])
        assert table.shape[0] == 4

    def test_zero_pivot_handling(self):
        # s^3 + s^2 + s + 1 zeros pivot
        table = routh_array([1, 1, 1, 1])
        assert table is not None


class TestCountRHPRoots:
    def test_stable_second_order(self):
        # s^2 + 2s + 1 → 0 RHP roots
        assert count_rhp_roots([1, 2, 1]) == 0

    def test_unstable(self):
        # s^2 - 1 → 1 RHP root
        assert count_rhp_roots([1, 0, -1]) >= 1

    def test_all_stable_3rd(self):
        # s^3 + 6s^2 + 11s + 6 = (s+1)(s+2)(s+3)
        assert count_rhp_roots([1, 6, 11, 6]) == 0

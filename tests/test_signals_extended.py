"""
Erweiterte Tests für pylabb.core.signals – 100 % Pfadabdeckung
"""
import numpy as np
import pytest

from pylabb.core.signals import (
    Signal,
    step_signal, ramp_signal, sine_signal, impulse_signal,
    square_signal, chirp_signal, noise_signal,
)


class TestSignalValidationErrors:
    def test_2d_t_raises(self):
        with pytest.raises(ValueError, match="eindimensional"):
            Signal(np.ones((2, 3)), np.ones(6))

    def test_2d_y_raises(self):
        with pytest.raises(ValueError, match="eindimensional"):
            Signal(np.arange(6, dtype=float), np.ones((2, 3)))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="gleich lang"):
            Signal(np.arange(5, dtype=float), np.arange(10, dtype=float))

    def test_single_sample_dt_defaults_to_1(self):
        s = Signal(np.array([0.0]), np.array([42.0]))
        assert s.dt == 1.0


class TestSignalArithmetic:
    def setup_method(self):
        t = np.linspace(0, 1, 50)
        self.t = t
        self.s1 = Signal(t, np.ones(50), name="A")
        self.s2 = Signal(t, np.ones(50) * 2, name="B")

    def test_add_signal(self):
        s3 = self.s1 + self.s2
        assert np.allclose(s3.y, 3.0)
        assert "A" in s3.name and "B" in s3.name

    def test_add_float(self):
        s3 = self.s1 + 5.0
        assert np.allclose(s3.y, 6.0)

    def test_sub_signal(self):
        s3 = self.s1 - self.s2
        assert np.allclose(s3.y, -1.0)
        assert "A" in s3.name and "B" in s3.name

    def test_sub_float(self):
        s3 = self.s1 - 0.5
        assert np.allclose(s3.y, 0.5)

    def test_mul_scalar(self):
        s3 = self.s1 * 3.0
        assert np.allclose(s3.y, 3.0)

    def test_rmul_scalar(self):
        s3 = 4.0 * self.s1
        assert np.allclose(s3.y, 4.0)

    def test_neg(self):
        s3 = -self.s1
        assert np.allclose(s3.y, -1.0)
        assert s3.name.startswith("-")


class TestSignalProperties:
    def test_fs_property(self):
        s = Signal(np.arange(1000) * 0.001, np.zeros(1000))
        assert abs(s.fs - 1000.0) < 0.01

    def test_duration(self):
        s = step_signal(t_end=5.0, dt=0.01)
        assert abs(s.duration - 5.0) < 0.02

    def test_n_samples(self):
        s = step_signal(t_end=1.0, dt=0.01)
        assert s.n_samples > 0

    def test_peak(self):
        t = np.linspace(0, 1, 100)
        s = Signal(t, np.sin(2 * np.pi * t))
        assert abs(s.peak() - 1.0) < 0.01

    def test_energy(self):
        t = np.linspace(0, 1, 10000)
        s = Signal(t, np.ones(10000))
        # energy = sum(y^2) * dt ≈ 1.0 s
        assert abs(s.energy() - 1.0) < 0.01

    def test_rms_sinusoid(self):
        t = np.linspace(0, 1, 10000, endpoint=False)
        s = Signal(t, np.sqrt(2) * np.sin(2 * np.pi * 10 * t))
        # RMS of A*sin = A/sqrt(2)
        assert abs(s.rms() - 1.0) < 0.01


class TestSignalMethods:
    def test_resample_fewer_samples(self):
        s = step_signal(t_end=1.0, dt=0.01)
        n_orig = s.n_samples
        s2 = s.resample(dt_new=0.02)
        assert s2.n_samples < n_orig
        assert abs(s2.t[1] - s2.t[0] - 0.02) < 1e-6

    def test_resample_preserves_name(self):
        s = step_signal(t_end=1.0, dt=0.01, name="TestSignal")
        s2 = s.resample(dt_new=0.05)
        assert s2.name == "TestSignal"

    def test_crop_units_preserved(self):
        t = np.linspace(0, 5, 500)
        s = Signal(t, np.ones(500), name="X", unit_t="ms", unit_y="V")
        sc = s.crop(1.0, 3.0)
        assert sc.unit_t == "ms"
        assert sc.unit_y == "V"
        assert sc.t[0] >= 1.0
        assert sc.t[-1] <= 3.0

    def test_repr(self):
        s = step_signal(t_end=2.0, dt=0.01, name="MySignal")
        r = repr(s)
        assert "MySignal" in r
        assert "Signal" in r
        assert "dt=" in r

    def test_fft_peak_at_correct_frequency(self):
        s = sine_signal(t_end=2.0, dt=0.0001, frequency=100.0)
        freqs, mags = s.fft()
        peak_freq = freqs[np.argmax(mags)]
        assert abs(peak_freq - 100.0) < 2.0


class TestFactoryFunctionsExtended:
    def test_step_delayed(self):
        s = step_signal(t_end=5.0, dt=0.01, t_step=2.5)
        before = s.y[s.t < 2.4]
        assert np.all(before == 0.0)
        after = s.y[s.t > 2.6]
        assert np.all(after > 0.0)

    def test_ramp_before_start_is_zero(self):
        s = ramp_signal(t_end=5.0, dt=0.01, t_start=2.0, slope=1.0)
        assert np.allclose(s.y[s.t < 1.9], 0.0)

    def test_sine_phase_offset(self):
        # phase=90° → sin(90°) = 1 at t=0
        s = sine_signal(t_end=1.0, dt=0.001, frequency=1.0, amplitude=1.0, phase_deg=90.0)
        assert abs(s.y[0] - 1.0) < 0.05

    def test_sine_dc_offset(self):
        s = sine_signal(t_end=1.0, dt=0.001, frequency=1.0, offset=3.0)
        assert abs(np.mean(s.y) - 3.0) < 0.05

    def test_impulse_delayed(self):
        s = impulse_signal(t_end=2.0, dt=0.01, t_impulse=1.0)
        idx = np.argmax(np.abs(s.y))
        assert abs(s.t[idx] - 1.0) < 0.02

    def test_impulse_unit_area(self):
        s = impulse_signal(t_end=1.0, dt=0.01, amplitude=1.0)
        area = np.sum(s.y) * s.dt
        assert abs(area - 1.0) < 0.01

    def test_square_signal_values(self):
        s = square_signal(t_end=2.0, dt=0.001, frequency=1.0,
                          amplitude=2.0, duty_cycle=0.5)
        assert s.n_samples > 0
        assert np.max(s.y) == pytest.approx(2.0, abs=0.01)

    def test_chirp_signal_amplitude(self):
        s = chirp_signal(t_end=2.0, dt=0.001, f0=1.0, f1=10.0, amplitude=3.0)
        assert abs(np.max(np.abs(s.y)) - 3.0) < 0.1

    def test_noise_mean_and_std(self):
        s = noise_signal(t_end=10.0, dt=0.001, std=2.0, mean=5.0, seed=0)
        assert abs(np.mean(s.y) - 5.0) < 0.2
        assert abs(np.std(s.y) - 2.0) < 0.2

    def test_noise_reproducible_with_seed(self):
        s1 = noise_signal(t_end=1.0, dt=0.001, seed=42)
        s2 = noise_signal(t_end=1.0, dt=0.001, seed=42)
        assert np.allclose(s1.y, s2.y)

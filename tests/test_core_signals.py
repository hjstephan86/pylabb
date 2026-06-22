"""
Tests für pylabb.core.signals
"""
import numpy as np
import pytest
from pylabb.core.signals import (
    Signal, step_signal, ramp_signal, sine_signal,
    impulse_signal, square_signal, chirp_signal, noise_signal,
)


class TestSignal:
    def test_creation(self):
        t = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * t)
        s = Signal(t, y, name="Sinus")
        assert s.n_samples == 100
        assert s.name == "Sinus"
        assert abs(s.duration - 1.0) < 1e-9

    def test_arithmetic(self):
        t = np.linspace(0, 1, 50)
        s1 = Signal(t, np.ones(50), name="A")
        s2 = Signal(t, np.ones(50) * 2, name="B")
        s3 = s1 + s2
        assert np.allclose(s3.y, 3.0)

    def test_rms(self):
        t = np.linspace(0, 1, 1000)
        s = Signal(t, np.ones(1000))
        assert abs(s.rms() - 1.0) < 1e-9

    def test_fft_returns_correct_shape(self):
        t = np.linspace(0, 1, 1000, endpoint=False)
        s = Signal(t, np.sin(2 * np.pi * 10 * t))
        freqs, mags = s.fft()
        assert freqs[0] == 0.0
        assert len(freqs) == len(mags)

    def test_crop(self):
        s = step_signal(t_end=5.0, dt=0.01)
        s_crop = s.crop(1.0, 3.0)
        assert s_crop.t[0] >= 1.0
        assert s_crop.t[-1] <= 3.0


class TestFactoryFunctions:
    def test_step(self):
        s = step_signal(t_end=5.0, dt=0.01, amplitude=2.0)
        assert s.y[-1] == pytest.approx(2.0)
        assert s.y[0] == pytest.approx(2.0)  # t_step=0 → sofortiger Sprung

    def test_ramp(self):
        s = ramp_signal(t_end=5.0, dt=0.01, slope=2.0)
        assert s.y[-1] == pytest.approx(5.0 * 2.0, rel=1e-3)

    def test_sine_frequency(self):
        s = sine_signal(t_end=1.0, dt=0.0001, frequency=5.0)
        freqs, mags = s.fft()
        peak_freq = freqs[np.argmax(mags)]
        assert abs(peak_freq - 5.0) < 0.5

    def test_noise_shape(self):
        s = noise_signal(t_end=2.0, dt=0.001, seed=42)
        assert len(s.y) > 0
        assert s.rms() > 0

    def test_chirp_length(self):
        s = chirp_signal(t_end=3.0, dt=0.001)
        assert s.n_samples > 0

"""
Tests für pylabb.visualization – 100 % Pfadabdeckung (nicht-GUI)
Matplotlib läuft auf dem Agg-Backend (kein Display erforderlich).
"""
import matplotlib
matplotlib.use("Agg")  # must be before any other matplotlib import

import numpy as np
import pytest
import matplotlib.pyplot as plt

from pylabb.core.transfer_function import TransferFunction
from pylabb.core.signals import Signal, step_signal
from pylabb.simulation.time_domain import SimulationResult

import pylabb.visualization as viz
from pylabb.visualization.bode import plot_bode
from pylabb.visualization.nyquist import plot_nyquist
from pylabb.visualization.rlocus import plot_root_locus
from pylabb.visualization.time_plots import (
    plot_step_response,
    plot_simulation,
    plot_spectrum,
    plot_pole_zero,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pt1():
    return TransferFunction([1], [1, 1], name="PT1")


def _pt2():
    return TransferFunction([1], [1, 0.5, 1], name="PT2")


def _sim_result():
    t = np.linspace(0, 10, 500)
    y = 1 - np.exp(-t)
    u = np.ones(500)
    e = 1 - y
    return SimulationResult(t=t, y=y, u=u, e=e, setpoint=np.ones(500), name="test")


# ---------------------------------------------------------------------------
# visualization __init__ re-exports
# ---------------------------------------------------------------------------

class TestVizInit:
    def test_all_symbols_importable(self):
        assert callable(viz.plot_bode)
        assert callable(viz.plot_nyquist)
        assert callable(viz.plot_root_locus)
        assert callable(viz.plot_step_response)
        assert callable(viz.plot_simulation)
        assert callable(viz.plot_spectrum)
        assert callable(viz.plot_pole_zero)


# ---------------------------------------------------------------------------
# Bode
# ---------------------------------------------------------------------------

class TestPlotBode:
    def test_single_system_returns_fig_axes(self):
        fig, (ax_mag, ax_phase) = plot_bode(_pt1())
        assert fig is not None
        assert ax_mag is not None
        assert ax_phase is not None
        plt.close(fig)

    def test_multiple_systems(self):
        fig, (ax_mag, ax_phase) = plot_bode(_pt1(), _pt2())
        assert fig is not None
        plt.close(fig)

    def test_show_margins_true(self):
        """Single system with show_margins=True triggers _draw_margins."""
        fig, axes = plot_bode(_pt1(), show_margins=True)
        assert fig is not None
        plt.close(fig)

    def test_show_margins_false(self):
        fig, axes = plot_bode(_pt1(), show_margins=False)
        plt.close(fig)

    def test_custom_omega(self):
        omega = np.logspace(-1, 2, 50)
        fig, axes = plot_bode(_pt1(), omega=omega)
        plt.close(fig)

    def test_no_grid(self):
        fig, axes = plot_bode(_pt1(), grid=False)
        plt.close(fig)

    def test_existing_figure_with_axes(self):
        """Pass a pre-existing Figure that already has 2 axes."""
        existing, _ = plt.subplots(2, 1)
        fig, axes = plot_bode(_pt1(), fig=existing)
        assert fig is existing
        plt.close(fig)

    def test_existing_figure_no_axes(self):
        """Pass a pre-existing Figure that has no axes."""
        existing = plt.figure()
        fig, axes = plot_bode(_pt1(), fig=existing)
        plt.close(fig)

    def test_multiple_systems_no_margins(self):
        """With multiple systems, _draw_margins is NOT called."""
        fig, _ = plot_bode(_pt1(), _pt2(), show_margins=True)
        plt.close(fig)

    def test_draw_margins_with_stable_system(self):
        """Second-order underdamped: finite phase-margin annotation."""
        G = TransferFunction([1], [1, 0.2, 1])
        fig, axes = plot_bode(G, show_margins=True)
        plt.close(fig)

    def test_draw_margins_gain_margin_annotated(self):
        """Third-order system has finite gain AND phase margins → both annotation paths."""
        # G(s) = 1 / (s+1)^3  has wpc = sqrt(3), gm = 8 → both finite
        G = TransferFunction([1], [1, 3, 3, 1])
        fig, axes = plot_bode(G, show_margins=True)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Nyquist
# ---------------------------------------------------------------------------

class TestPlotNyquist:
    def test_basic(self):
        fig, ax = plot_nyquist(_pt1())
        assert fig is not None
        plt.close(fig)

    def test_multiple_systems(self):
        fig, ax = plot_nyquist(_pt1(), _pt2())
        plt.close(fig)

    def test_no_unit_circle(self):
        fig, ax = plot_nyquist(_pt1(), show_unit_circle=False)
        plt.close(fig)

    def test_with_unit_circle(self):
        fig, ax = plot_nyquist(_pt1(), show_unit_circle=True)
        plt.close(fig)

    def test_custom_omega(self):
        omega = np.logspace(-1, 2, 100)
        fig, ax = plot_nyquist(_pt1(), omega=omega)
        plt.close(fig)

    def test_existing_figure_with_axes(self):
        existing, ax_existing = plt.subplots()
        fig, ax = plot_nyquist(_pt1(), fig=existing)
        assert fig is existing
        plt.close(fig)

    def test_existing_figure_no_axes(self):
        existing = plt.figure()
        fig, ax = plot_nyquist(_pt1(), fig=existing)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Root Locus
# ---------------------------------------------------------------------------

class TestPlotRootLocus:
    def test_basic(self):
        G = TransferFunction([1], [1, 3, 2])
        fig, ax = plot_root_locus(G)
        assert fig is not None
        plt.close(fig)

    def test_no_open_loop_markers(self):
        G = TransferFunction([1], [1, 3, 2])
        fig, ax = plot_root_locus(G, show_open_loop=False)
        plt.close(fig)

    def test_system_with_zeros(self):
        """Triggers the zeros branch in show_open_loop."""
        G = TransferFunction([1, 2], [1, 3, 2])
        fig, ax = plot_root_locus(G, show_open_loop=True)
        plt.close(fig)

    def test_custom_k_range(self):
        G = TransferFunction([1], [1, 2, 1])
        k_range = np.linspace(0.1, 10, 50)
        fig, ax = plot_root_locus(G, k_range=k_range)
        plt.close(fig)

    def test_existing_figure_with_axes(self):
        G = TransferFunction([1], [1, 2, 1])
        existing, _ = plt.subplots()
        fig, ax = plot_root_locus(G, fig=existing)
        assert fig is existing
        plt.close(fig)

    def test_existing_figure_no_axes(self):
        G = TransferFunction([1], [1, 2, 1])
        existing = plt.figure()
        fig, ax = plot_root_locus(G, fig=existing)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Step response
# ---------------------------------------------------------------------------

class TestPlotStepResponse:
    def test_basic(self):
        fig, ax = plot_step_response(_pt1())
        assert fig is not None
        plt.close(fig)

    def test_multiple_systems(self):
        """Multiple systems: _annotate_step is NOT called."""
        fig, ax = plot_step_response(_pt1(), _pt2())
        plt.close(fig)

    def test_no_characteristics(self):
        fig, ax = plot_step_response(_pt1(), show_characteristics=False)
        plt.close(fig)

    def test_annotate_step_with_overshoot(self):
        """Underdamped system → annotate_step draws both settling time and overshoot."""
        G = TransferFunction([1], [1, 0.2, 1])
        fig, ax = plot_step_response(G, show_characteristics=True, t_end=30.0)
        plt.close(fig)

    def test_annotate_step_no_overshoot(self):
        """Overdamped system → annotate_step draws settling time only."""
        G = TransferFunction([1], [1, 3, 1])
        fig, ax = plot_step_response(G, show_characteristics=True)
        plt.close(fig)

    def test_existing_figure_with_axes(self):
        existing, _ = plt.subplots()
        fig, ax = plot_step_response(_pt1(), fig=existing)
        assert fig is existing
        plt.close(fig)

    def test_existing_figure_no_axes(self):
        existing = plt.figure()
        fig, ax = plot_step_response(_pt1(), fig=existing)
        plt.close(fig)

    def test_custom_t_end(self):
        fig, ax = plot_step_response(_pt1(), t_end=3.0, dt=0.01)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Simulation dashboard
# ---------------------------------------------------------------------------

class TestPlotSimulation:
    def test_basic(self):
        fig, axes = plot_simulation(_sim_result())
        assert fig is not None
        assert len(axes) == 3
        plt.close(fig)

    def test_existing_figure(self):
        existing = plt.figure()
        fig, axes = plot_simulation(_sim_result(), fig=existing)
        assert fig is existing
        plt.close(fig)

    def test_custom_title(self):
        fig, axes = plot_simulation(_sim_result(), title="Mein Test")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

class TestPlotSpectrum:
    def _signal(self):
        t = np.linspace(0, 2.0, 1000)
        y = np.sin(2 * np.pi * 5 * t)
        return Signal(t, y, name="sine5Hz")

    def test_log_scale(self):
        fig, ax = plot_spectrum(self._signal(), log_scale=True)
        assert fig is not None
        plt.close(fig)

    def test_linear_scale(self):
        fig, ax = plot_spectrum(self._signal(), log_scale=False)
        plt.close(fig)

    def test_multiple_signals(self):
        s1 = self._signal()
        t = np.linspace(0, 2.0, 1000)
        s2 = Signal(t, np.sin(2 * np.pi * 10 * t), name="sine10Hz")
        fig, ax = plot_spectrum(s1, s2)
        plt.close(fig)

    def test_existing_figure_with_axes(self):
        existing, _ = plt.subplots()
        fig, ax = plot_spectrum(self._signal(), fig=existing)
        plt.close(fig)

    def test_existing_figure_no_axes(self):
        existing = plt.figure()
        fig, ax = plot_spectrum(self._signal(), fig=existing)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Pole-zero
# ---------------------------------------------------------------------------

class TestPlotPoleZero:
    def test_basic(self):
        fig, ax = plot_pole_zero(_pt1())
        assert fig is not None
        plt.close(fig)

    def test_system_with_zeros(self):
        """Triggers the zeros plotting branch."""
        G = TransferFunction([1, 1], [1, 3, 2])
        fig, ax = plot_pole_zero(G)
        plt.close(fig)

    def test_no_zeros(self):
        G = TransferFunction([1], [1, 2, 1])
        fig, ax = plot_pole_zero(G)
        plt.close(fig)

    def test_show_unit_circle_true(self):
        fig, ax = plot_pole_zero(_pt1(), show_unit_circle=True)
        plt.close(fig)

    def test_show_unit_circle_false(self):
        fig, ax = plot_pole_zero(_pt1(), show_unit_circle=False)
        plt.close(fig)

    def test_multiple_systems(self):
        fig, ax = plot_pole_zero(_pt1(), _pt2())
        plt.close(fig)

    def test_existing_figure_with_axes(self):
        existing, _ = plt.subplots()
        fig, ax = plot_pole_zero(_pt1(), fig=existing)
        assert fig is existing
        plt.close(fig)

    def test_existing_figure_no_axes(self):
        existing = plt.figure()
        fig, ax = plot_pole_zero(_pt1(), fig=existing)
        plt.close(fig)

"""
pylabb.visualization.time_plots
================================
Zeitbereichsplots: Sprungantwort, Stellgröße, Regelfehler, FFT-Spektrum,
Pol-Nullstellen-Diagramm.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.gridspec as gridspec

from pylabb.core.signals import Signal
from pylabb.core.transfer_function import TransferFunction


# ---------------------------------------------------------------------------
# Sprungantwort
# ---------------------------------------------------------------------------

def plot_step_response(
    *systems: TransferFunction,
    t_end: Optional[float] = None,
    dt: float = 0.005,
    title: str = "Sprungantwort",
    figsize: tuple[float, float] = (9, 5),
    show_characteristics: bool = True,
    fig: Optional[matplotlib.figure.Figure] = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Sprungantwort(en) im Zeitbereich.

    Parameters
    ----------
    *systems              : TransferFunction-Objekte.
    show_characteristics  : Zeigt Einschwingzeit, Überschwingen etc.
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        axes = fig.get_axes()
        ax = axes[0] if axes else fig.add_subplot(1, 1, 1)

    ax.set_title(title)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, G in enumerate(systems):
        color = colors[idx % len(colors)]
        t, y = G.step_response(t_end=t_end, dt=dt)
        label = G.name or f"G{idx+1}"
        ax.plot(t, y, color=color, label=label, linewidth=1.8)

        if show_characteristics and len(systems) == 1:
            _annotate_step(ax, t, y, color)

    ax.set_xlabel("Zeit t [s]")
    ax.set_ylabel("y(t)")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.7, alpha=0.6, label="Sollwert")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig, ax


def _annotate_step(
    ax: matplotlib.axes.Axes, t: NDArray, y: NDArray, color: str
) -> None:
    """Beschriftet charakteristische Punkte der Sprungantwort."""
    from pylabb.simulation.time_domain import SimulationResult
    res = SimulationResult(
        t=t, y=y, u=np.zeros_like(y), e=1.0 - y, setpoint=np.ones_like(y)
    )
    ts = res.settling_time()
    os_pct = res.overshoot_pct()

    if np.isfinite(ts) and ts > 0:
        y_at_ts = np.interp(ts, t, y)
        ax.axvline(ts, color=color, linestyle=":", linewidth=0.9, alpha=0.7)
        ax.annotate(
            f"ts={ts:.2f}s",
            xy=(ts, y_at_ts),
            xytext=(ts + 0.1 * t[-1], y_at_ts * 0.8),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.7),
        )

    if np.isfinite(os_pct) and os_pct > 0.5:
        t_peak = t[np.argmax(y)]
        y_peak = np.max(y)
        ax.plot(t_peak, y_peak, "o", color=color, markersize=5)
        ax.annotate(
            f"Ü={os_pct:.1f}%",
            xy=(t_peak, y_peak),
            xytext=(t_peak + 0.05 * t[-1], y_peak * 1.02),
            fontsize=8, color=color,
        )


# ---------------------------------------------------------------------------
# Simulation-Dashboard (y, u, e in einem Plot)
# ---------------------------------------------------------------------------

def plot_simulation(
    result: "SimulationResult",
    figsize: tuple[float, float] = (10, 9),
    title: str = "Regleranalyse",
    fig: Optional[matplotlib.figure.Figure] = None,
) -> tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, ...]]:
    """Dreifach-Dashboard: Ausgang, Stellgröße und Regelfehler.

    Parameters
    ----------
    result : ``SimulationResult`` aus ``simulation.time_domain``.
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    fig.suptitle(f"{title} – {result.name}")

    # Ausgang + Sollwert
    ax1.plot(result.t, result.setpoint, "k--", linewidth=0.9, label="Sollwert w")
    ax1.plot(result.t, result.y, color="C0", linewidth=1.8, label="Ausgang y")
    ax1.set_ylabel("y, w")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Stellgröße
    ax2.plot(result.t, result.u, color="C1", linewidth=1.8)
    ax2.set_ylabel("Stellgröße u")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Regelfehler
    ax3.plot(result.t, result.e, color="C2", linewidth=1.8)
    ax3.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax3.set_ylabel("Fehler e = w − y")
    ax3.set_xlabel("Zeit t [s]")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Kennwerte
    s = result.summary()
    info = (
        f"ts={s['Einschwingzeit [s]']:.3g}s  "
        f"Ü={s['Überschwingen [%]']:.2g}%  "
        f"IAE={s['IAE']:.3g}  "
        f"e∞={s['Stationäre Abweichung']:.3g}"
    )
    fig.text(0.5, 0.01, info, ha="center", fontsize=9, style="italic")
    # subplots_adjust works correctly with GridSpec (tight_layout raises UserWarning)
    fig.subplots_adjust(top=0.93, bottom=0.08, hspace=0.45)
    return fig, (ax1, ax2, ax3)


# ---------------------------------------------------------------------------
# FFT-Betragsspektrum
# ---------------------------------------------------------------------------

def plot_spectrum(
    *signals: Signal,
    title: str = "Frequenzspektrum",
    figsize: tuple[float, float] = (9, 5),
    log_scale: bool = True,
    fig: Optional[matplotlib.figure.Figure] = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """FFT-Betragsspektrum von Signalen.

    Parameters
    ----------
    *signals   : Signal-Objekte.
    log_scale  : X-Achse logarithmisch.
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        axes = fig.get_axes()
        ax = axes[0] if axes else fig.add_subplot(1, 1, 1)

    ax.set_title(title)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, s in enumerate(signals):
        freqs, mags = s.fft()
        color = colors[idx % len(colors)]
        label = s.name or f"Signal {idx+1}"
        if log_scale:
            ax.semilogy(freqs, mags, color=color, label=label, linewidth=1.5)
        else:
            ax.plot(freqs, mags, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Frequenz [Hz]")
    ax.set_ylabel("Betrag")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Pol-Nullstellen-Diagramm
# ---------------------------------------------------------------------------

def plot_pole_zero(
    *systems: TransferFunction,
    title: str = "Pol-Nullstellen-Diagramm",
    figsize: tuple[float, float] = (8, 7),
    show_unit_circle: bool = False,
    fig: Optional[matplotlib.figure.Figure] = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Pol-Nullstellen-Diagramm (s-Ebene / z-Ebene).

    Pole: ×  |  Nullstellen: ○
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        axes = fig.get_axes()
        ax = axes[0] if axes else fig.add_subplot(1, 1, 1)

    ax.set_title(title)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, G in enumerate(systems):
        color = colors[idx % len(colors)]
        label = G.name or f"G{idx+1}"
        poles = G.poles()
        zeros = G.zeros()

        ax.plot(poles.real, poles.imag, "x",
                color=color, markersize=10, markeredgewidth=2, label=f"{label} Pole")
        if len(zeros):
            ax.plot(zeros.real, zeros.imag, "o",
                    color=color, markersize=8, fillstyle="none",
                    markeredgewidth=2, label=f"{label} Nullstellen")

    if show_unit_circle:
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.7, alpha=0.5, label="Einheitskreis")

    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5, label="jω-Achse")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Import aus simulation (verzögert, um zirkuläre Importe zu vermeiden)
# ---------------------------------------------------------------------------
try:
    from pylabb.simulation.time_domain import SimulationResult  # noqa: F401
except ImportError:  # pragma: no cover
    SimulationResult = None  # type: ignore

"""
pylabb.visualization.bode
=========================
Bode-Diagramm (Amplituden- und Phasengang, optional mit Stabilitätsrändern).
Unterstützt mehrere Übertragungsfunktionen in einem Plot (Overlay).
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes

from pylabb.core.transfer_function import TransferFunction


def plot_bode(
    *systems: TransferFunction,
    omega: Optional[NDArray] = None,
    n_points: int = 1000,
    show_margins: bool = True,
    title: str = "Bode-Diagramm",
    figsize: tuple[float, float] = (10, 7),
    grid: bool = True,
    fig: Optional[matplotlib.figure.Figure] = None,
) -> tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """Bode-Diagramm für eine oder mehrere ÜTF.

    Parameters
    ----------
    *systems     : Übertragungsfunktionen (variadisch).
    omega        : Kreisfrequenzbereich [rad/s]; None → automatisch.
    n_points     : Anzahl Frequenzpunkte.
    show_margins : Zeichnet Amplituden- und Phasenrand ein.
    title        : Fenstertitel.
    figsize      : Plot-Größe.
    grid         : Gitterlinien.
    fig          : Bestehendes Figure-Objekt (zum Einbetten in GUI).

    Returns
    -------
    fig, (ax_mag, ax_phase)
    """
    if fig is None:
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        axes = fig.get_axes()
        if len(axes) >= 2:
            ax_mag, ax_phase = axes[0], axes[1]
        else:
            ax_mag = fig.add_subplot(2, 1, 1)
            ax_phase = fig.add_subplot(2, 1, 2, sharex=ax_mag)

    fig.suptitle(title)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, G in enumerate(systems):
        color = colors[idx % len(colors)]
        o, mag_dB, phase_deg = G.bode(omega=omega, n_points=n_points)

        label = G.name or f"G{idx+1}"
        ax_mag.semilogx(o, mag_dB, color=color, label=label, linewidth=1.8)
        ax_phase.semilogx(o, phase_deg, color=color, label=label, linewidth=1.8)

        if show_margins and len(systems) == 1:
            _draw_margins(ax_mag, ax_phase, o, mag_dB, phase_deg)

    # Achsen-Beschriftung
    ax_mag.set_ylabel("Amplitude [dB]")
    ax_mag.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax_mag.legend(loc="lower left")

    ax_phase.set_ylabel("Phase [°]")
    ax_phase.set_xlabel("Kreisfrequenz ω [rad/s]")
    ax_phase.axhline(-180, color="k", linestyle="--", linewidth=0.8)
    ax_phase.legend(loc="lower left")

    if grid:
        ax_mag.grid(True, which="both", linestyle=":", alpha=0.7)
        ax_phase.grid(True, which="both", linestyle=":", alpha=0.7)

    fig.tight_layout()
    return fig, (ax_mag, ax_phase)


def _draw_margins(
    ax_mag: matplotlib.axes.Axes,
    ax_phase: matplotlib.axes.Axes,
    omega: NDArray,
    mag_dB: NDArray,
    phase_deg: NDArray,
) -> None:
    """Zeichnet Amplitunden- und Phasenrand als Pfeile/Linien ein."""
    from pylabb.control.stability import _bode_margins
    gm, pm, wgc, wpc = _bode_margins(omega, mag_dB, phase_deg)

    # Durchtrittsfrequenz (Phasenrand)
    if np.isfinite(wgc) and np.isfinite(pm):
        ph_at_gco = np.interp(np.log10(wgc), np.log10(omega), phase_deg)
        ax_mag.axvline(wgc, color="red", linestyle=":", linewidth=0.9, alpha=0.7)
        ax_phase.axvline(wgc, color="red", linestyle=":", linewidth=0.9, alpha=0.7)
        ax_phase.annotate(
            f"ΦR = {pm:.1f}°",
            xy=(wgc, ph_at_gco),
            xytext=(wgc * 1.5, ph_at_gco + 20),
            color="red", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

    # Phasenumkehrfrequenz (Amplitudenrand)
    if np.isfinite(wpc) and np.isfinite(gm):
        mag_at_pco = np.interp(np.log10(wpc), np.log10(omega), mag_dB)
        ax_mag.axvline(wpc, color="blue", linestyle=":", linewidth=0.9, alpha=0.7)
        ax_phase.axvline(wpc, color="blue", linestyle=":", linewidth=0.9, alpha=0.7)
        ax_mag.annotate(
            f"AR = {gm:.1f} dB",
            xy=(wpc, mag_at_pco),
            xytext=(wpc * 1.5, mag_at_pco + 5),
            color="blue", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="blue", lw=0.8),
        )

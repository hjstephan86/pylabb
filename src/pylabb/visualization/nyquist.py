"""
pylabb.visualization.nyquist
============================
Nyquist-Ortskurve mit optionaler Hervorhebung des kritischen Punkts
(-1+j0) und Pfeilen für die Umlaufrichtung.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.patches as mpatches

from pylabb.core.transfer_function import TransferFunction


def plot_nyquist(
    *systems: TransferFunction,
    omega: Optional[NDArray] = None,
    n_points: int = 2000,
    show_unit_circle: bool = True,
    title: str = "Nyquist-Ortskurve",
    figsize: tuple[float, float] = (8, 7),
    fig: Optional[matplotlib.figure.Figure] = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Nyquist-Diagramm einer oder mehrerer Übertragungsfunktionen.

    Zeichnet sowohl positiven (ω > 0) als auch negativen Ast (ω < 0)
    und markiert den kritischen Punkt −1+j0.

    Parameters
    ----------
    *systems         : ÜTF-Objekte.
    omega            : Kreisfrequenzvektor [rad/s]; None → automatisch.
    n_points         : Anzahl Frequenzpunkte.
    show_unit_circle : Einheitskreis als Referenz.
    title, figsize   : Plotoptionen.
    fig              : Bestehendes Figure (für GUI-Einbettung).

    Returns
    -------
    fig, ax
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

        if omega is None:
            from pylabb.core.transfer_function import _auto_omega
            o = _auto_omega(G, n_points)
        else:
            o = omega

        _, H = G.freqresp(o)

        # Positiver Ast
        ax.plot(H.real, H.imag, color=color, label=label, linewidth=1.8)
        # Negativer Ast (konjugiert komplex)
        ax.plot(H.real, -H.imag, color=color, linestyle="--", linewidth=1.0, alpha=0.5)

        # Pfeil zur Kennzeichnung des Umlaufsinns
        mid = len(o) // 2
        ax.annotate(
            "",
            xy=(H.real[mid + 1], H.imag[mid + 1]),
            xytext=(H.real[mid], H.imag[mid]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    # Kritischer Punkt
    ax.plot(-1, 0, "rx", markersize=12, markeredgewidth=2, label="−1+j0")

    if show_unit_circle:
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), "k:", linewidth=0.6, alpha=0.4, label="Einheitskreis")

    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Re{G(jω)}")
    ax.set_ylabel("Im{G(jω)}")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig, ax

"""
pylabb.visualization.rlocus
===========================
Wurzelortskurve (Root Locus): Pol-Pfade des geschlossenen Kreises
bei variierender Verstärkung K.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.stability import root_locus_data


def plot_root_locus(
    G: TransferFunction,
    k_range: Optional[NDArray] = None,
    n_k: int = 1000,
    title: str = "Wurzelortskurve",
    figsize: tuple[float, float] = (9, 7),
    show_open_loop: bool = True,
    fig: Optional[matplotlib.figure.Figure] = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Zeichnet die Wurzelortskurve von G(s).

    Parameters
    ----------
    G              : Offene Kreisübertragungsfunktion.
    k_range        : Verstärkungsvektor; None → logspace(-2, 3, n_k).
    n_k            : Schritte (wenn k_range=None).
    show_open_loop : Polt offene Pole (×) und Nullstellen (○).
    title, figsize : Plotoptionen.
    fig            : Bestehendes Figure.

    Returns
    -------
    fig, ax
    """
    k_vec, rl_poles = root_locus_data(G, k_range=k_range, n_k=n_k)

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        axes = fig.get_axes()
        ax = axes[0] if axes else fig.add_subplot(1, 1, 1)

    ax.set_title(title)

    n_poles = rl_poles.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_poles))

    for i in range(n_poles):
        branch = rl_poles[:, i]
        ax.plot(branch.real, branch.imag, color=colors[i], linewidth=1.5,
                label=f"Pol {i+1}")

    if show_open_loop:
        ol_poles = G.poles()
        ol_zeros = G.zeros()
        ax.plot(ol_poles.real, ol_poles.imag, "rx",
                markersize=10, markeredgewidth=2, label="OL-Pole")
        if len(ol_zeros):
            ax.plot(ol_zeros.real, ol_zeros.imag, "bo",
                    markersize=8, fillstyle="none", markeredgewidth=2, label="OL-Nullstellen")

    # Stabilitätsgrenze
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.6, label="jω-Achse")

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig, ax

"""
pylabb.control.stability
=======================
Stabilitätsanalyse für LTI-Systeme (Linear Time-Invariant).

Enthält:
- Routh-Hurwitz-Kriterium
- Nyquist-Kriterium (vereinfachte Version)
- Gain-/Phasenrand aus dem Bode-Diagramm
- Pol-Nullstellen-Analyse
- Dominante Pole und Zeitkonstanten
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from pylabb.core.transfer_function import TransferFunction
from pylabb.core.math_utils import routh_array, count_rhp_roots


# ---------------------------------------------------------------------------
# Stabilitätseigenschaften als Datenklasse
# ---------------------------------------------------------------------------

@dataclass
class StabilityInfo:
    """Zusammenfassung aller Stabilitätskenngrößen.

    Attributes
    ----------
    is_stable           : True, wenn das OL-System stabil ist.
    gain_margin_dB      : Amplitudenrand [dB].
    phase_margin_deg    : Phasenrand [°].
    gain_crossover_freq : Durchtrittsfrequenz (|G(jω)| = 0 dB) [rad/s].
    phase_crossover_freq: Phasenumkehrfrequenz (∠G(jω) = -180°) [rad/s].
    rhp_poles           : Anzahl instabiler Pole.
    poles               : Alle Pole der Übertragungsfunktion.
    zeros               : Alle Nullstellen.
    dominant_poles      : Dominante Pole (kleinste negative Realteile).
    """
    is_stable: bool
    gain_margin_dB: float
    phase_margin_deg: float
    gain_crossover_freq: float
    phase_crossover_freq: float
    rhp_poles: int
    poles: NDArray
    zeros: NDArray
    dominant_poles: NDArray

    def __str__(self) -> str:
        lines = [
            "=== Stabilitätsanalyse ===",
            f"Stabil:              {self.is_stable}",
            f"Instabile Pole:      {self.rhp_poles}",
            f"Amplitudenrand:      {self.gain_margin_dB:.2f} dB",
            f"Phasenrand:          {self.phase_margin_deg:.2f}°",
            f"Durchtrittsfrequenz: {self.gain_crossover_freq:.4g} rad/s",
            f"Phasenumkehrfreq.:   {self.phase_crossover_freq:.4g} rad/s",
            f"Pole: {np.array2string(self.poles, precision=4, suppress_small=True)}",
            f"Nullstellen: {np.array2string(self.zeros, precision=4, suppress_small=True)}",
            f"Dominante Pole: {np.array2string(self.dominant_poles, precision=4, suppress_small=True)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hauptanalyse-Funktion
# ---------------------------------------------------------------------------

def analyze(G: TransferFunction, n_points: int = 5000) -> StabilityInfo:
    """Führt eine vollständige Stabilitätsanalyse des offenen Regelkreises
    (Übertragungsfunktion G) durch.

    Parameters
    ----------
    G        : Offene Kreisübertragungsfunktion.
    n_points : Anzahl der Frequenzpunkte für Bode-Analyse.

    Returns
    -------
    StabilityInfo mit allen berechneten Kenngrößen.
    """
    poles = G.poles()
    zeros = G.zeros()

    # Routh-basierte Stabilitätsprüfung (nur für kont. Systeme)
    if G.dt is None:
        rhp = count_rhp_roots(G.den)
        is_stable = rhp == 0
    else:
        rhp = int(np.sum(np.abs(poles) >= 1.0))
        is_stable = rhp == 0

    omega, mag_dB, phase_deg = G.bode(n_points=n_points)

    gm_dB, pm_deg, wgc, wpc = _bode_margins(omega, mag_dB, phase_deg)

    dom_poles = _dominant_poles(poles)

    return StabilityInfo(
        is_stable=is_stable,
        gain_margin_dB=gm_dB,
        phase_margin_deg=pm_deg,
        gain_crossover_freq=wgc,
        phase_crossover_freq=wpc,
        rhp_poles=rhp,
        poles=poles,
        zeros=zeros,
        dominant_poles=dom_poles,
    )


# ---------------------------------------------------------------------------
# Bode-Stabilitätskriterium
# ---------------------------------------------------------------------------

def bode_margins(
    G: TransferFunction,
    n_points: int = 5000,
) -> tuple[float, float, float, float]:
    """Berechnet Amplitudenrand und Phasenrand aus dem Bode-Diagramm.

    Returns
    -------
    gm_dB   : Amplitudenrand [dB]  (∞ wenn keine Phasendurchgang).
    pm_deg  : Phasenrand [°]       (∞ wenn kein Phasendurchgang).
    wgc     : Durchtrittsfrequenz  [rad/s].
    wpc     : Phasenumkehrfrequenz [rad/s].
    """
    omega, mag_dB, phase_deg = G.bode(n_points=n_points)
    return _bode_margins(omega, mag_dB, phase_deg)


def _bode_margins(
    omega: NDArray, mag_dB: NDArray, phase_deg: NDArray
) -> tuple[float, float, float, float]:
    """Interne Berechnung der Stabilitätsränder."""
    # Durchtrittsfrequenz ωgc: |G(jω)| = 0 dB
    idx_cross = np.where(np.diff(np.sign(mag_dB)))[0]
    if len(idx_cross):
        # lineare Interpolation
        i = idx_cross[0]
        frac = -mag_dB[i] / (mag_dB[i + 1] - mag_dB[i])
        wgc = omega[i] + frac * (omega[i + 1] - omega[i])
        phase_at_gco = phase_deg[i] + frac * (phase_deg[i + 1] - phase_deg[i])
        pm_deg = 180.0 + phase_at_gco
    else:
        wgc = float("inf")
        pm_deg = float("inf")

    # Phasenumkehrfrequenz ωpc: ∠G(jω) = -180°
    phase_shifted = phase_deg + 180.0
    idx_phase = np.where(np.diff(np.sign(phase_shifted)))[0]
    if len(idx_phase):
        i = idx_phase[0]
        frac = -phase_shifted[i] / (phase_shifted[i + 1] - phase_shifted[i])
        wpc = omega[i] + frac * (omega[i + 1] - omega[i])
        mag_at_pco = mag_dB[i] + frac * (mag_dB[i + 1] - mag_dB[i])
        gm_dB = -mag_at_pco
    else:
        wpc = float("inf")
        gm_dB = float("inf")

    return float(gm_dB), float(pm_deg), float(wgc), float(wpc)


# ---------------------------------------------------------------------------
# Pol-Nullstellen-Analyse
# ---------------------------------------------------------------------------

def _dominant_poles(poles: NDArray, n: int = 2) -> NDArray:
    """Gibt die *n* dominanten Pole zurück (kleinste negative Realteile)."""
    stable_poles = poles[np.real(poles) < 0]
    if len(stable_poles) == 0:
        return poles
    sorted_poles = stable_poles[np.argsort(np.abs(np.real(stable_poles)))]
    return sorted_poles[:n]


def pole_zero_info(G: TransferFunction) -> dict:
    """Übersichtliches Dictionary mit Pol-/Nullstelleninformationen."""
    poles = G.poles()
    zeros = G.zeros()
    return {
        "poles": poles,
        "zeros": zeros,
        "natural_frequencies": np.abs(poles),
        "damping_ratios": -np.real(poles) / np.abs(poles),
        "is_minimum_phase": bool(np.all(np.real(zeros) < 0)),
    }


# ---------------------------------------------------------------------------
# Wurzelortskurven-Daten
# ---------------------------------------------------------------------------

def root_locus_data(
    G: TransferFunction,
    k_range: Optional[NDArray] = None,
    n_k: int = 500,
) -> tuple[NDArray, NDArray]:
    """Berechnet die Pole des geschlossenen Regelkreises für variierende
    Verstärkung K.

    T(s) = K·G(s) / (1 + K·G(s))

    Parameters
    ----------
    G       : Offener Kreis G(s).
    k_range : Verstärkungsvektor; wenn None → logspace(−2, 3, n_k).
    n_k     : Anzahl Verstärkungsschritte.

    Returns
    -------
    k_range : Verstärkungsvektor [n_k].
    rl_poles: Polpfade [n_k × Systemordnung] als komplexes Array.
    """
    if k_range is None:
        k_range = np.logspace(-2, 3, n_k)

    n_poles = len(G.den) - 1
    rl_poles = np.zeros((len(k_range), n_poles), dtype=complex)

    for i, k in enumerate(k_range):
        closed_den = np.polyadd(G.den, k * G.num)
        rl_poles[i] = np.roots(closed_den)

    return k_range, rl_poles


def critical_gain(G: TransferFunction) -> tuple[float, float]:
    """Bestimmt die kritische Verstärkung Ku und die Schwingungsfrequenz ωu
    anhand der Bode-Analyse (marginale Stabilität).

    Returns
    -------
    Ku, wu   : Kritische Verstärkung, Schwingungskreisfrequenz [rad/s].
    """
    omega, H = G.freqresp()
    phase_deg = np.rad2deg(np.unwrap(np.angle(H)))
    mag = np.abs(H)

    idx = np.argmin(np.abs(phase_deg + 180.0))
    wu = float(omega[idx])
    Ku = float(1.0 / max(mag[idx], 1e-12))
    return Ku, wu

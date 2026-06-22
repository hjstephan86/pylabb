"""
pylabb.control.design
====================
Methoden zum Reglerentwurf:

* Polvorgabe für Zustandsrückführung          (→ StateSpace.place_poles)
* LQR-Entwurf                                 (→ StateSpace.lqr)
* Lead/Lag-Kompensatoren
* Notch-/Bandpassfilter als ÜTF
* Loopshaping-Hilfsfunktionen
* Fehlerstrom- und Führungsübertragungsfunktionen
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray

from pylabb.core.transfer_function import TransferFunction
from pylabb.core.state_space import StateSpace


# ---------------------------------------------------------------------------
# Lead-/Lag-Kompensator
# ---------------------------------------------------------------------------

def lead_compensator(
    alpha: float,          # α > 1 für Lead, 0 < α < 1 für Lag
    T: float,              # Zeitkonstante [s]
    K: float = 1.0,
) -> TransferFunction:
    """Erstellt einen Lead/Lag-Kompensator.

    C(s) = K · (T·s + 1) / (α·T·s + 1)

    Parameters
    ----------
    alpha : > 1 → Lead-Kompensator (Phasenanhebung)
            < 1 → Lag-Kompensator  (I-Näherung)
    T     : Eckfrequenz 1/T [rad/s].
    K     : Skalierungsverstärkung.

    Returns
    -------
    TransferFunction
    """
    name = "Lead" if alpha > 1 else "Lag"
    num = [K * T, K]
    den = [alpha * T, 1]
    return TransferFunction(num, den, name=name)


def lag_compensator(
    alpha: float,
    T: float,
    K: float = 1.0,
) -> TransferFunction:
    """Lag-Kompensator: führt low-frequency gain hinzu ohne Phasenverlust.

    Entspricht lead_compensator(alpha < 1, T, K).
    """
    assert alpha < 1, "Für Lag-Kompensator muss α < 1 sein."
    return lead_compensator(alpha, T, K)


def lead_lag_compensator(
    T1: float, alpha1: float,
    T2: float, alpha2: float,
    K: float = 1.0,
) -> TransferFunction:
    """Kombinierter Lead-Lag-Kompensator (Reihenschaltung)."""
    L = lead_compensator(alpha1, T1, K)
    Lag = lead_compensator(alpha2, T2, 1.0)
    return L * Lag


# ---------------------------------------------------------------------------
# Notch- / Bandpass-Filter als ÜTF
# ---------------------------------------------------------------------------

def notch_filter(
    omega_n: float,
    zeta_z: float = 0.01,
    zeta_p: float = 0.3,
    K: float = 1.0,
) -> TransferFunction:
    """Kerbfilter (Notch) zur Unterdrückung einer Störfrequenz ωn.

    C(s) = K · (s² + 2ζ_z·ωn·s + ωn²) / (s² + 2ζ_p·ωn·s + ωn²)

    Parameters
    ----------
    omega_n : Kerbfrequenz [rad/s].
    zeta_z  : Dämpfungsgrad der Nullstellen (< zeta_p → Kerbe).
    zeta_p  : Dämpfungsgrad der Pole.
    """
    num = [K, 2 * zeta_z * omega_n * K, omega_n**2 * K]
    den = [1, 2 * zeta_p * omega_n, omega_n**2]
    return TransferFunction(num, den, name=f"Notch({omega_n:.3g})")


def bandpass_filter(
    omega_c: float,
    Q: float = 10.0,
    K: float = 1.0,
) -> TransferFunction:
    """Bandpassfilter um ωc mit Güte Q.

    C(s) = K · (ω_c/Q·s) / (s² + ω_c/Q·s + ωc²)
    """
    bw = omega_c / Q
    num = [K * bw, 0]
    den = [1, bw, omega_c**2]
    return TransferFunction(num, den, name=f"BP({omega_c:.3g})")


def lowpass_filter(
    omega_c: float,
    order: int = 1,
    K: float = 1.0,
) -> TransferFunction:
    """Butterworthtiefpass der Ordnung 1 oder 2.

    Parameters
    ----------
    omega_c : Eckfrequenz [rad/s].
    order   : 1 oder 2.
    """
    if order == 1:
        return TransferFunction([K * omega_c], [1, omega_c], name="LP1")
    elif order == 2:
        # Butterworth: ζ = 1/√2
        zeta = 1.0 / np.sqrt(2)
        num = [K * omega_c**2]
        den = [1, 2 * zeta * omega_c, omega_c**2]
        return TransferFunction(num, den, name="LP2")
    else:
        raise ValueError("Nur Ordnung 1 oder 2 unterstützt.")


def highpass_filter(omega_c: float, K: float = 1.0) -> TransferFunction:
    """Einfaches Hochpassfilter erster Ordnung."""
    return TransferFunction([K, 0], [1, omega_c], name="HP1")


# ---------------------------------------------------------------------------
# Geschlossener Regelkreis / Sensitivitätsfunktionen
# ---------------------------------------------------------------------------

def closed_loop(
    G: TransferFunction,
    C: TransferFunction,
    H: Optional[TransferFunction] = None,
) -> TransferFunction:
    """Berechnet die Führungsübertragungsfunktion T(s) = C·G / (1 + C·G·H).

    Parameters
    ----------
    G : Strecke.
    C : Regler.
    H : Messglied (None → H=1).
    """
    if H is None:
        H = TransferFunction([1], [1], dt=G.dt)
    L = C * G * H  # Kreisübertragungsfunktion
    one = TransferFunction([1], [1], dt=G.dt)
    return (C * G) / (one + L)


def sensitivity(
    G: TransferFunction,
    C: TransferFunction,
) -> TransferFunction:
    """Sensitivitätsfunktion S(s) = 1 / (1 + L(s))."""
    L = C * G
    one = TransferFunction([1], [1], dt=G.dt)
    return one / (one + L)


def complementary_sensitivity(
    G: TransferFunction,
    C: TransferFunction,
) -> TransferFunction:
    """Komplementäre Sensitivität T(s) = L(s) / (1 + L(s))."""
    L = C * G
    one = TransferFunction([1], [1], dt=G.dt)
    return L / (one + L)


def control_sensitivity(
    G: TransferFunction,
    C: TransferFunction,
) -> TransferFunction:
    """Stellgrößen-Sensitivität  Q(s) = C(s) / (1 + L(s))."""
    L = C * G
    one = TransferFunction([1], [1], dt=G.dt)
    return C / (one + L)


# ---------------------------------------------------------------------------
# Loopshaping-Hilfsfunktionen
# ---------------------------------------------------------------------------

def gain_to_dB(K: float) -> float:
    """Linearverstärkung → dB."""
    return 20 * np.log10(abs(K))


def dB_to_gain(dB: float) -> float:
    """dB → Linearverstärkung."""
    return 10 ** (dB / 20)


def series(*transfer_functions: TransferFunction) -> TransferFunction:
    """Reihenschalten beliebig vieler ÜTF: G = G1 * G2 * … * Gn."""
    result = transfer_functions[0]
    for G in transfer_functions[1:]:
        result = result * G
    return result


def parallel(*transfer_functions: TransferFunction) -> TransferFunction:
    """Parallelschalten beliebig vieler ÜTF: G = G1 + G2 + … + Gn."""
    result = transfer_functions[0]
    for G in transfer_functions[1:]:
        result = result + G
    return result


# ---------------------------------------------------------------------------
# Totzeitapproximation (Padé)
# ---------------------------------------------------------------------------

def pade_approximation(L: float, order: int = 2) -> TransferFunction:
    """Padé-Approximation der Totzeit e^{-Ls}.

    Parameters
    ----------
    L     : Totzeit [s].
    order : Ordnung der Approximation (1..4).

    Returns
    -------
    TransferFunction der rationalen Approximation.
    """
    if order == 1:
        num = [-L / 2, 1]
        den = [L / 2, 1]
    elif order == 2:
        num = [L**2 / 12, -L / 2, 1]
        den = [L**2 / 12, L / 2, 1]
    elif order == 3:
        num = [-L**3 / 120, L**2 / 10, -L / 2, 1]
        den = [L**3 / 120, L**2 / 10, L / 2, 1]
    elif order == 4:
        num = [L**4 / 1680, -L**3 / 60, L**2 / 7, -L / 2, 1]
        den = [L**4 / 1680, L**3 / 60, L**2 / 7, L / 2, 1]
    else:
        raise ValueError("Ordnung 1–4 unterstützt.")
    return TransferFunction(num, den, name=f"Pade({L:.3g},n={order})")


# ---------------------------------------------------------------------------
# Standard-Streckenmodelle
# ---------------------------------------------------------------------------

def first_order_plant(K: float, T: float, L: float = 0.0) -> TransferFunction:
    """FOPDT-Streckennäherung K·e^{-Ls} / (Ts+1).

    Die Totzeit wird als Padé-Approximation 2. Ordnung realisiert.

    Parameters
    ----------
    K : Statische Verstärkung.
    T : Zeitkonstante [s].
    L : Totzeit [s].
    """
    G_p = TransferFunction([K], [T, 1], name="P1")
    if L <= 0:
        return G_p
    return G_p * pade_approximation(L, order=2)


def second_order_plant(
    K: float, omega_n: float, zeta: float, L: float = 0.0
) -> TransferFunction:
    """PT2-Strecke K·ωn² / (s² + 2ζωns + ωn²) optional mit Totzeit."""
    G_p = TransferFunction(
        [K * omega_n**2],
        [1, 2 * zeta * omega_n, omega_n**2],
        name="P2",
    )
    if L <= 0:
        return G_p
    return G_p * pade_approximation(L, order=2)


def integrating_plant(K: float, T: float = 0.0) -> TransferFunction:
    """Integrierend-behaftete Strecke: K/s oder K/(s*(Ts+1))."""
    if T <= 0:
        return TransferFunction([K], [1, 0], name="I")
    return TransferFunction([K], [T, 1, 0], name="IT1")

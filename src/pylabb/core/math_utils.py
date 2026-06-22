"""
pylabb.core.math_utils
=====================
Grundlegende mathematische Hilfsfunktionen für die Regelungstechnik.

Enthält Polynomoperationen, Partialbruchzerlegung, bilineare Transformation
(Tustin-Methode), Zero-Order-Hold-Diskretisierung und Routh-Hurwitz-Array.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Polynomfunktionen
# ---------------------------------------------------------------------------

def poly_eval(coeffs: Sequence[float], x: complex | np.ndarray) -> complex | np.ndarray:
    """Wertet ein Polynom p(x) = a0*x^n + a1*x^(n-1) + … + an aus.

    Parameters
    ----------
    coeffs : Koeffizienten in absteigender Reihenfolge (höchste Potenz zuerst).
    x      : Auswertungspunkt(e).

    Returns
    -------
    Wert des Polynoms an x.

    Examples
    --------
    >>> poly_eval([1, -3, 2], 1.0)   # x^2 - 3x + 2 bei x=1 → 0
    0.0
    """
    return np.polyval(coeffs, x)


def poly_multiply(p: Sequence[float], q: Sequence[float]) -> np.ndarray:
    """Multipliziert zwei Polynome (Faltung der Koeffizienten)."""
    return np.polymul(p, q)


def poly_add(p: Sequence[float], q: Sequence[float]) -> np.ndarray:
    """Addiert zwei Polynome, Längenausgleich durch Nullauffüllung."""
    return np.polyadd(p, q)


def roots_of_polynomial(coeffs: Sequence[float]) -> np.ndarray:
    """Berechnet alle Nullstellen eines Polynoms."""
    return np.roots(coeffs)


# ---------------------------------------------------------------------------
# Partialbruchzerlegung
# ---------------------------------------------------------------------------

def partial_fraction(
    num: Sequence[float],
    den: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partialbruchzerlegung H(s) = num(s)/den(s).

    Returns
    -------
    residues : Zähler der Partialbrüche.
    poles    : Pole.
    gain     : Direktanteil (bei unecht gebrochenem Bruch).

    Notes
    -----
    Nutzt :func:`scipy.signal.residue` intern.
    """
    from scipy.signal import residue
    return residue(num, den)


# ---------------------------------------------------------------------------
# Bilineare Transformation (Tustin/Bilinear z-Domain)
# ---------------------------------------------------------------------------

def bilinear_transform(
    num: Sequence[float],
    den: Sequence[float],
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wandelt eine kontinuierliche Übertragungsfunktion H(s) per Tustin-
    Methode in eine zeitdiskrete H(z) um.

    Parameters
    ----------
    num : Zählerkoeffizienten von H(s).
    den : Nennerkoeffizienten von H(s).
    fs  : Abtastfrequenz [Hz].

    Returns
    -------
    num_d, den_d : Zähler/Nenner der diskreten Übertragungsfunktion H(z).
    """
    from scipy.signal import bilinear
    return bilinear(num, den, fs=fs)


# ---------------------------------------------------------------------------
# Zero-Order-Hold-Diskretisierung
# ---------------------------------------------------------------------------

def zoh_discretize(
    A: np.ndarray,
    B: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Diskretisiert ein zeitkontinuierliches Zustandsraumsystem
    ẋ = Ax + Bu per Zero-Order-Hold.

    Parameters
    ----------
    A  : Systemmatrix (n×n).
    B  : Eingangsmatrix (n×m).
    dt : Abtastzeit [s].

    Returns
    -------
    Ad, Bd : Diskrete Systemmatrix und Eingangsmatrix.

    Notes
    -----
    Verwendet :func:`scipy.linalg.expm` für die Matrixexponentialfunktion.
    """
    from scipy.linalg import expm

    n = A.shape[0]
    m = B.shape[1] if B.ndim == 2 else 1

    # Erweiterungsmatrix für genaue ZOH-Berechnung
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n:] = B * dt
    eM = expm(M)

    Ad = eM[:n, :n]
    Bd = eM[:n, n:]
    return Ad, Bd


# ---------------------------------------------------------------------------
# Routh-Hurwitz-Array
# ---------------------------------------------------------------------------

def routh_array(coeffs: Sequence[float]) -> np.ndarray:
    """Berechnet das Routh-Array für das charakteristische Polynom.

    Parameters
    ----------
    coeffs : Koeffizienten in absteigender Reihenfolge (a_n, …, a_0).

    Returns
    -------
    routh : 2D-Array (n+1 Zeilen).

    Raises
    ------
    ValueError : Bei inkonsistenter Eingabe.

    Examples
    --------
    >>> routh_array([1, 2, 3, 4])   # s^3 + 2s^2 + 3s + 4
    """
    coeffs = np.array(coeffs, dtype=float)
    n = len(coeffs)

    rows = n
    cols = int(np.ceil(n / 2))
    table = np.zeros((rows, cols))

    # Erste zwei Zeilen aus den Koeffizienten
    table[0, :len(coeffs[0::2])] = coeffs[0::2]
    table[1, :len(coeffs[1::2])] = coeffs[1::2]

    eps = 1e-12
    for i in range(2, rows):
        for j in range(cols - 1):
            pivot = table[i - 1, 0]
            if abs(pivot) < eps:
                pivot = eps  # Sonderfall: Nullzeile
            table[i, j] = (
                table[i - 1, 0] * table[i - 2, j + 1]
                - table[i - 2, 0] * table[i - 1, j + 1]
            ) / pivot

    return table


def count_rhp_roots(coeffs: Sequence[float]) -> int:
    """Zählt die Vorzeichen­wechsel in der ersten Spalte des Routh-Arrays,
    was der Anzahl der instabilen Pole (rechte Halbebene) entspricht."""
    table = routh_array(coeffs)
    first_col = table[:, 0]
    sign_changes = int(np.sum(np.diff(np.sign(first_col)) != 0))
    return sign_changes

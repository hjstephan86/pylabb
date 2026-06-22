"""
pylabb.core.transfer_function
=============================
Übertragungsfunktion G(s) = B(s)/A(s) in Laplace-Darstellung.

Unterstützt:
- Algebraische Operationen (+, *, /, Serienschaltung, Parallelschaltung)
- Frequenzgangberechnung G(jω)
- Berechnung von Polen und Nullstellen
- Zeitantworten (Sprung, Impuls) via scipy.signal
- Konvertierung in Zustandsraumdarstellung
- Serielle Diskretisierung (ZOH, Tustin)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import scipy.signal as sig

if TYPE_CHECKING:
    from pylabb.core.state_space import StateSpace


class TransferFunction:
    """Lineare, zeitinvariante Übertragungsfunktion im Laplace-/z-Bereich.

    Parameters
    ----------
    num   : Zählerkoeffizienten   [a_n, a_{n-1}, …, a_0].
    den   : Nennerkoeffizienten   [b_m, b_{m-1}, …, b_0].
    dt    : Abtastzeit > 0 für zeitdiskrete Systeme (None → kontinuierlich).
    name  : Kurzbezeichnung (z. B. 'G_P').

    Examples
    --------
    >>> G = TransferFunction([1], [1, 2, 1])   # 1/(s²+2s+1) = 1/(s+1)²
    >>> G.poles()
    array([-1., -1.])
    """

    def __init__(
        self,
        num: Sequence[float],
        den: Sequence[float],
        dt: Optional[float] = None,
        name: str = "G",
    ) -> None:
        self.num = np.atleast_1d(np.asarray(num, dtype=complex))
        self.den = np.atleast_1d(np.asarray(den, dtype=complex))
        self.dt = dt
        self.name = name
        self._normalize()

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _normalize(self) -> None:
        """Entfernt führende Nullen und stellt Reell-Darstellung sicher."""
        trimmed_num = np.trim_zeros(self.num, trim="f")
        self.num = trimmed_num if len(trimmed_num) > 0 else np.array([0.0])
        trimmed_den = np.trim_zeros(self.den, trim="f")
        self.den = trimmed_den if len(trimmed_den) > 0 else np.array([1.0])
        if np.all(np.isreal(self.num)):
            self.num = self.num.real
        if np.all(np.isreal(self.den)):
            self.den = self.den.real

    def _as_scipy(self) -> sig.TransferFunction:
        """Gibt das scipy-Pendant zurück."""
        if self.dt is None:
            return sig.TransferFunction(self.num, self.den)
        return sig.dlti(self.num, self.den, dt=self.dt)

    # ------------------------------------------------------------------
    # Frequenzgang
    # ------------------------------------------------------------------

    def freqresp(
        self,
        omega: Optional[NDArray] = None,
        n_points: int = 500,
    ) -> tuple[NDArray, NDArray]:
        """Berechnet den Frequenzgang G(jω).

        Parameters
        ----------
        omega    : Kreisfrequenzen [rad/s]; wenn None, automatisch gewählt.
        n_points : Anzahl Punkte (nur wenn omega=None).

        Returns
        -------
        omega, H : Kreisfrequenzen und komplexe Übertragungswerte.
        """
        if omega is None:
            omega = _auto_omega(self, n_points)
        if self.dt is None:
            H = np.polyval(self.num, 1j * omega) / np.polyval(self.den, 1j * omega)
        else:
            z = np.exp(1j * omega * self.dt)
            H = np.polyval(self.num, z) / np.polyval(self.den, z)
        return omega, H

    def bode(
        self,
        omega: Optional[NDArray] = None,
        n_points: int = 500,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Bode-Daten: Amplitude in dB und Phase in Grad.

        Returns
        -------
        omega, mag_dB, phase_deg
        """
        omega, H = self.freqresp(omega, n_points)
        mag_dB = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
        phase_deg = np.rad2deg(np.unwrap(np.angle(H)))
        return omega, mag_dB, phase_deg

    # ------------------------------------------------------------------
    # Polstellen / Nullstellen
    # ------------------------------------------------------------------

    def poles(self) -> NDArray:
        """Pole der Übertragungsfunktion."""
        return np.roots(self.den)

    def zeros(self) -> NDArray:
        """Nullstellen der Übertragungsfunktion."""
        return np.roots(self.num)

    def is_stable(self) -> bool:
        """True, wenn alle Pole in der linken Halbebene (kont.) oder
        im Einheitskreis (diskret) liegen."""
        p = self.poles()
        if self.dt is None:
            return bool(np.all(np.real(p) < 0))
        return bool(np.all(np.abs(p) < 1.0))

    def dc_gain(self) -> complex:
        """Stationäre Verstärkung G(0)."""
        if self.dt is None:
            return complex(np.polyval(self.num, 0) / np.polyval(self.den, 0))
        return complex(np.polyval(self.num, 1) / np.polyval(self.den, 1))

    # ------------------------------------------------------------------
    # Zeitantworten
    # ------------------------------------------------------------------

    def step_response(
        self, t_end: Optional[float] = None, dt: Optional[float] = None
    ) -> tuple[NDArray, NDArray]:
        """Sprungantwort h(t).

        Returns
        -------
        t, y
        """
        if self.dt is not None:
            return self._step_discrete()
        t_arr = _sim_time(self, t_end=t_end, dt=dt)
        t_out, y_out = sig.step(self._as_scipy(), T=t_arr)
        return t_out, y_out

    def _step_discrete(self) -> tuple[NDArray, NDArray]:
        n = 200
        system = sig.dlti(self.num, self.den, dt=self.dt)
        t_arr = np.arange(n) * self.dt
        t_out, y_out = sig.dstep(system, t=t_arr)
        return t_out, np.squeeze(y_out)

    def impulse_response(
        self, t_end: Optional[float] = None, dt: Optional[float] = None
    ) -> tuple[NDArray, NDArray]:
        """Impulsantwort g(t) via scipy.

        Returns
        -------
        t, y
        """
        if self.dt is not None:
            system = sig.dlti(self.num, self.den, dt=self.dt)
            t_arr = np.arange(200) * self.dt
            t_out, y_out = sig.dimpulse(system, t=t_arr)
            return t_out, np.squeeze(y_out)
        t_arr = _sim_time(self, t_end=t_end, dt=dt)
        t_out, y_out = sig.impulse(self._as_scipy(), T=t_arr)
        return t_out, y_out

    def lsim(
        self,
        u: NDArray,
        t: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Lineare Systemsimulation mit beliebigem Eingang u(t).

        Parameters
        ----------
        u : Eingangssignal (1-D oder 2-D).
        t : Zeitachse [s].

        Returns
        -------
        t, y
        """
        if self.dt is None:
            t_out, y_out, _ = sig.lsim(self._as_scipy(), U=u, T=t)
        else:
            system = sig.dlti(self.num, self.den, dt=self.dt)
            t_out, y_out = sig.dlsim(system, u=u.reshape(-1, 1), t=t)
            y_out = y_out.flatten()
        return t_out, y_out

    # ------------------------------------------------------------------
    # Konvertimmierung
    # ------------------------------------------------------------------

    def to_state_space(self) -> "StateSpace":
        """Konvertiert in Zustandsraumdarstellung."""
        from pylabb.core.state_space import StateSpace as _SS
        A, B, C, D = sig.tf2ss(self.num, self.den)
        return _SS(A, B, C, D, dt=self.dt, name=self.name)

    def discretize(
        self, dt: float, method: str = "tustin"
    ) -> "TransferFunction":
        """Diskretisierung der kontinuierlichen ÜTF.

        Parameters
        ----------
        dt     : Abtastzeit [s].
        method : 'tustin' (Standard) | 'zoh' | 'euler'
        """
        if self.dt is not None:
            raise ValueError("System ist bereits zeitdiskret.")
        if method in ("tustin", "bilinear"):
            num_d, den_d = sig.bilinear(self.num, self.den, fs=1.0 / dt)
        elif method == "zoh":
            sys_d = sig.cont2discrete(
                (self.num, self.den), dt=dt, method="zoh"
            )
            num_d, den_d = sys_d[0].flatten(), sys_d[1]
        elif method == "euler":
            sys_d = sig.cont2discrete(
                (self.num, self.den), dt=dt, method="euler"
            )
            num_d, den_d = sys_d[0].flatten(), sys_d[1]
        else:
            raise ValueError(f"Unbekannte Methode: {method!r}")
        return TransferFunction(num_d, den_d, dt=dt, name=self.name + "_d")

    # ------------------------------------------------------------------
    # Algebraische Operationen
    # ------------------------------------------------------------------

    def __mul__(self, other: "TransferFunction | float") -> "TransferFunction":
        """Reihenschaltung G1 * G2."""
        if isinstance(other, (int, float)):
            return TransferFunction(
                np.polymul(self.num, [other]), self.den, dt=self.dt
            )
        _check_compat(self, other)
        return TransferFunction(
            np.polymul(self.num, other.num),
            np.polymul(self.den, other.den),
            dt=self.dt,
        )

    def __rmul__(self, scalar: float) -> "TransferFunction":
        return self.__mul__(scalar)

    def __truediv__(self, other: "TransferFunction | float") -> "TransferFunction":
        if isinstance(other, (int, float)):
            return TransferFunction(self.num, np.polymul(self.den, [other]), dt=self.dt)
        _check_compat(self, other)
        return TransferFunction(
            np.polymul(self.num, other.den),
            np.polymul(self.den, other.num),
            dt=self.dt,
        )

    def __add__(self, other: "TransferFunction | float") -> "TransferFunction":
        """Parallelschaltung."""
        if isinstance(other, (int, float)):
            other = TransferFunction([other], [1], dt=self.dt)
        _check_compat(self, other)
        return TransferFunction(
            np.polyadd(
                np.polymul(self.num, other.den),
                np.polymul(other.num, self.den),
            ),
            np.polymul(self.den, other.den),
            dt=self.dt,
        )

    def __radd__(self, other: float) -> "TransferFunction":
        return self.__add__(other)

    def __sub__(self, other: "TransferFunction | float") -> "TransferFunction":
        if isinstance(other, (int, float)):
            other = TransferFunction([other], [1], dt=self.dt)
        return self.__add__(-1.0 * other)

    def __neg__(self) -> "TransferFunction":
        return TransferFunction(-self.num, self.den, dt=self.dt, name=f"-{self.name}")

    def feedback(
        self, H: Optional["TransferFunction"] = None, sign: int = -1
    ) -> "TransferFunction":
        """Geschlossener Regelkreis T(s) = G / (1 ∓ G·H).

        Parameters
        ----------
        H    : Rückführungsübertragungsfunktion (None → H=1).
        sign : -1 (Gegenkopplung, Standard) | +1 (Mitkopplung).
        """
        if H is None:
            H = TransferFunction([1], [1], dt=self.dt)
        _check_compat(self, H)
        # Direkte Formel: T = num_G * den_H / (den_G * den_H - sign * num_G * num_H)
        # Vermeidet ungekürzte gemeinsame Faktoren beim intermediären Schritt
        num_T = np.polymul(self.num, H.den)
        den_T = np.polyadd(
            np.polymul(self.den, H.den),
            np.polymul((-sign * self.num), H.num),
        )
        return TransferFunction(num_T, den_T, dt=self.dt)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        num_str = np.array2string(self.num, precision=4, suppress_small=True)
        den_str = np.array2string(self.den, precision=4, suppress_small=True)
        disc = f", dt={self.dt}" if self.dt else ""
        return f"TransferFunction(num={num_str}, den={den_str}{disc}, name={self.name!r})"


# ---------------------------------------------------------------------------
# Hilfsfunktionen (modulintern)
# ---------------------------------------------------------------------------

def _check_compat(a: TransferFunction, b: TransferFunction) -> None:
    if a.dt != b.dt:
        raise ValueError(
            "Abtastzeiten der Übertragungsfunktionen stimmen nicht überein: "
            f"{a.dt} vs {b.dt}"
        )


def _auto_omega(G: TransferFunction, n: int) -> NDArray:
    """Wählt automatisch einen sinnvollen Omega-Bereich."""
    p = np.abs(G.poles())
    z = np.abs(G.zeros())
    freqs = np.concatenate([p[p > 0], z[z > 0]])
    if len(freqs) == 0:
        return np.logspace(-2, 2, n)
    w_min = max(np.min(freqs) / 100, 1e-3)
    w_max = np.max(freqs) * 100
    return np.logspace(np.log10(w_min), np.log10(w_max), n)


def _sim_time(
    G: TransferFunction,
    t_end: Optional[float] = None,
    dt: Optional[float] = None,
) -> NDArray:
    """Erstellt eine Standardzeitachse für Simulationen."""
    p = G.poles()
    real_parts = np.real(p[np.real(p) < 0])
    if len(real_parts):
        tau = -1.0 / np.max(real_parts)
        t_end = t_end or min(max(10 * tau, 1.0), 100.0)
    else:
        t_end = t_end or 10.0
    dt = dt or t_end / 2000
    return np.linspace(0, t_end, int(t_end / dt) + 1)

"""
pylabb.control.pid
=================
PID-Regler (Proportional-Integral-Differential) in verschiedenen
Ausprägungen:

* ``PIDController``          – Idealer PID mit kontinuierlicher ÜTF
* ``DiscretePIDController``  – Zeitdiskreter PID (Velocity- oder
                               Positions-Algorithmus, Anti-Windup)
* Entwurfshilfen:
  - ``ziegler_nichols_step``  – Schwingungsversuch (kontinuierlich)
  - ``ziegler_nichols_relay`` – Relaismethode (vereinfacht)
  - ``cohen_coon``            – Regler nach Cohen-Coon (FOPDT-Modell)
  - ``lambda_tuning``         – λ-Einstellung (FOPDT-Modell)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np
from numpy.typing import NDArray

from pylabb.core.transfer_function import TransferFunction


# ---------------------------------------------------------------------------
# Kontinuierlicher PID-Regler
# ---------------------------------------------------------------------------

@dataclass
class PIDController:
    """Idealer PID-Regler mit Tiefpassfilter auf dem D-Anteil.

    Übertragungsfunktion::

        C(s) = Kp * (1 + 1/(Ti*s) + Td*s) * 1/(1 + s*Td/N)

    Parameters
    ----------
    Kp : Proportionalverstärkung.
    Ti : Nachstellzeit [s]  (0 = kein I-Anteil).
    Td : Vorhaltzeit   [s]  (0 = kein D-Anteil).
    N  : Filterkonstante D-Glied (Standard: 20).
    name : Bezeichnung.
    """
    Kp: float = 1.0
    Ti: float = 0.0
    Td: float = 0.0
    N:  float = 20.0
    name: str = "PID"

    def transfer_function(self) -> TransferFunction:
        """Gibt die kontinuierliche Übertragungsfunktion C(s) zurück."""
        Kp, Ti, Td, N = self.Kp, self.Ti, self.Td, self.N

        # P-Regler
        if Ti == 0 and Td == 0:
            return TransferFunction([Kp], [1], name=self.name)

        # PI-Regler
        if Td == 0:
            # C(s) = Kp*(1 + 1/(Ti*s)) = Kp*(Ti*s + 1)/(Ti*s)
            return TransferFunction([Kp * Ti, Kp], [Ti, 0], name=self.name)

        # D-Filter: 1/(1 + Td/N * s)
        d_filter_den = [Td / N, 1.0]

        if Ti == 0:
            # PD: C(s) = Kp*(1 + Td*s) / (1 + Td/N*s)
            num = np.polymul([Kp * Td, Kp], [1])
            den = d_filter_den
            # Normalisierung
            num, den = np.polymul(num, [1]), np.polymul(den, [1])
            return TransferFunction(num, den, name=self.name)

        # Vollständiger PID:  C(s) = Kp*(Ti*Td*s² + Ti*s + 1) / (Ti*s*(1+Td/N*s))
        # Zähler = Kp*(Ti*Td*s² + Ti*s + 1)
        pid_num = [Kp * Ti * Td, Kp * Ti, Kp]
        # Nenner = Ti * [Td/N, 1] * s = Ti * [Td/N, 1, 0]
        pid_den = np.polymul([Ti, 0], d_filter_den)

        # Zähler ebenfalls mit s multiplizieren (schon im Nenner durch *s)
        # => C(s) = pid_num / pid_den
        return TransferFunction(pid_num, pid_den, name=self.name)

    def __repr__(self) -> str:
        return (
            f"PIDController(Kp={self.Kp:.4g}, Ti={self.Ti:.4g}, "
            f"Td={self.Td:.4g}, N={self.N:.4g}, name={self.name!r})"
        )


# ---------------------------------------------------------------------------
# Zeitdiskreter PID-Regler (für Simulation und MicroPython-Codegen)
# ---------------------------------------------------------------------------

class DiscretePIDController:
    """Zeitdiskreter PID-Regler (Positions- und Velocity-Algorithmus).

    Implementierung nach dem ISA-Standard mit Anti-Windup (Rückverfolgung)
    und Stellgrößenbegrenzung.

    Parameters
    ----------
    Kp          : Proportionalverstärkung.
    Ti          : Nachstellzeit [s]  (inf → kein I-Anteil).
    Td          : Vorhaltzeit [s].
    dt          : Abtastzeit [s].
    u_min/u_max : Stellgrößengrenzen.
    N           : D-Filter-Koeffizient.
    algorithm   : 'position' | 'velocity'.
    anti_windup : Aktiviert Anti-Windup (Rückverfolgungskonstante = Ti).
    """

    def __init__(
        self,
        Kp: float = 1.0,
        Ti: float = float("inf"),
        Td: float = 0.0,
        dt: float = 0.01,
        u_min: float = -1e9,
        u_max: float = 1e9,
        N: float = 20.0,
        algorithm: Literal["position", "velocity"] = "position",
        anti_windup: bool = True,
    ) -> None:
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.N = N
        self.algorithm = algorithm
        self.anti_windup = anti_windup
        self.reset()

    def reset(self) -> None:
        """Setzt alle internen Zustände zurück."""
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._prev_deriv: float = 0.0
        self._prev_u: float = 0.0

    def update(self, setpoint: float, measurement: float) -> float:
        """Berechnet einen Regelschritt und gibt die Stellgröße zurück.

        Parameters
        ----------
        setpoint    : Sollwert.
        measurement : Istwert (gemessener Ausgang).

        Returns
        -------
        u : Stellgröße (geclippt auf [u_min, u_max]).
        """
        error = setpoint - measurement

        # --- Proportionalanteil ---
        P = self.Kp * error

        # --- Integralanteil ---
        if self.Ti != float("inf") and self.Ti > 0:
            self._integral += error * self.dt
            I = self.Kp / self.Ti * self._integral
        else:
            I = 0.0

        # --- Differentialanteil (mit Tiefpassfilter) ---
        if self.Td > 0:
            alpha = self.Td / (self.Td + self.N * self.dt)
            deriv = alpha * self._prev_deriv + (1 - alpha) * (error - self._prev_error) / self.dt
            D = self.Kp * self.Td * deriv
            self._prev_deriv = deriv
        else:
            D = 0.0

        u_raw = P + I + D
        u = float(np.clip(u_raw, self.u_min, self.u_max))

        # --- Anti-Windup ---
        if self.anti_windup and self.Ti > 0 and self.Ti != float("inf"):
            windup = (u - u_raw) / (self.Kp / self.Ti) if self.Kp != 0 else 0.0
            self._integral += windup * self.dt

        self._prev_error = error
        self._prev_u = u
        return u

    def simulate(
        self,
        setpoints: NDArray,
        t: Optional[NDArray] = None,
    ) -> tuple[NDArray, NDArray]:
        """Simuliert eine offene Sequenz von Sollwertvorgaben ohne Rückkopplung.

        Nützlich zum Testen des Reglers mit synthetischen Eingaben.

        Parameters
        ----------
        setpoints : 1-D Array der Sollwerte.
        t         : optionale Zeitachse; wenn None → linspace.

        Returns
        -------
        t, u : Ausgabesequenz der Stellgrößen.
        """
        self.reset()
        n = len(setpoints)
        u_arr = np.zeros(n)
        t_arr = t if t is not None else np.arange(n) * self.dt
        measurement = 0.0
        for i, sp in enumerate(setpoints):
            u_arr[i] = self.update(sp, measurement)
        return t_arr, u_arr

    def to_params(self) -> dict:
        """Serialisiert die PID-Parameter als Dictionary (für Codegen)."""
        return {
            "Kp": self.Kp, "Ti": self.Ti, "Td": self.Td, "dt": self.dt,
            "u_min": self.u_min, "u_max": self.u_max, "N": self.N,
        }

    def __repr__(self) -> str:
        return (
            f"DiscretePIDController(Kp={self.Kp}, Ti={self.Ti}, "
            f"Td={self.Td}, dt={self.dt})"
        )


# ---------------------------------------------------------------------------
# Einstellregeln
# ---------------------------------------------------------------------------

def ziegler_nichols_step(
    K: float, T: float, L: float,
    controller_type: Literal["P", "PI", "PID"] = "PID",
) -> PIDController:
    """Berechnet PID-Parameter nach Ziegler-Nichols (Sprungantwortmethode).

    Das Streckenmodell wird als PTt-Glied (FOPDT) angenähert:
      G_s(s) ≈ K · e^{-Ls} / (1 + Ts)

    Parameters
    ----------
    K             : Statische Streckenverstärkung.
    T             : Zeitkonstante [s].
    L             : Totzeit/Verzugszeit [s].
    controller_type : 'P', 'PI' oder 'PID'.

    Returns
    -------
    PIDController mit berechneten Parametern.
    """
    if controller_type == "P":
        Kp = T / (K * L)
        return PIDController(Kp=Kp, Ti=0.0, Td=0.0, name="P-ZN")
    elif controller_type == "PI":
        Kp = 0.9 * T / (K * L)
        Ti = L / 0.3
        return PIDController(Kp=Kp, Ti=Ti, Td=0.0, name="PI-ZN")
    else:  # PID
        Kp = 1.2 * T / (K * L)
        Ti = 2.0 * L
        Td = 0.5 * L
        return PIDController(Kp=Kp, Ti=Ti, Td=Td, name="PID-ZN")


def ziegler_nichols_oscillation(
    Ku: float,
    Tu: float,
    controller_type: Literal["P", "PI", "PID"] = "PID",
) -> PIDController:
    """Berechnet PID-Parameter aus kritischer Verstärkung und Periodendauer
    (Schwingungsversuch / Ultimate Cycle Method).

    Parameters
    ----------
    Ku : Kritische Verstärkung (System beginnt zu schwingen).
    Tu : Kritische Periode [s].
    controller_type : 'P', 'PI' oder 'PID'.
    """
    if controller_type == "P":
        return PIDController(Kp=0.5 * Ku, Ti=0.0, Td=0.0, name="P-ZN-Osc")
    elif controller_type == "PI":
        return PIDController(Kp=0.45 * Ku, Ti=Tu / 1.2, Td=0.0, name="PI-ZN-Osc")
    else:
        return PIDController(
            Kp=0.6 * Ku, Ti=Tu / 2.0, Td=Tu / 8.0, name="PID-ZN-Osc"
        )


def cohen_coon(
    K: float, T: float, L: float,
    controller_type: Literal["P", "PI", "PD", "PID"] = "PID",
) -> PIDController:
    """PID-Einstellung nach Cohen-Coon für FOPDT-Modelle.

    Parameters
    ----------
    K, T, L : Streckenparameter (Verstärkung, Zeitkonstante, Totzeit).
    """
    tau = L / T  # normierte Totzeit
    if controller_type == "P":
        Kp = (1 / (K * tau)) * (1 + tau / 3)
        return PIDController(Kp=Kp, Ti=0.0, Td=0.0, name="P-CC")
    elif controller_type == "PI":
        Kp = (1 / (K * tau)) * (0.9 + tau / 12)
        Ti = L * (30 + 3 * tau) / (9 + 20 * tau)
        return PIDController(Kp=Kp, Ti=Ti, Td=0.0, name="PI-CC")
    elif controller_type == "PD":
        Kp = (1 / (K * tau)) * (1.24 + tau * 0.267)
        Td = L * (6 - 2 * tau) / (22 + 3 * tau)
        return PIDController(Kp=Kp, Ti=0.0, Td=Td, name="PD-CC")
    else:  # PID
        Kp = (1 / (K * tau)) * (4 / 3 + tau / 4)
        Ti = L * (32 + 6 * tau) / (13 + 8 * tau)
        Td = L * 4 / (11 + 2 * tau)
        return PIDController(Kp=Kp, Ti=Ti, Td=Td, name="PID-CC")


def lambda_tuning(
    K: float, T: float, L: float,
    lambda_: Optional[float] = None,
) -> PIDController:
    """λ-Einstellung (IMC-basiert) für FOPDT-Modelle.

    Parameters
    ----------
    K, T, L  : Streckenparameter.
    lambda_  : Closed-loop-Zeitkonstante [s]; Standard: max(0.5*T, 2.0*L).

    Returns
    -------
    PI-Regler.
    """
    if lambda_ is None:
        lambda_ = max(0.5 * T, 2.0 * L)
    Kp = T / (K * (lambda_ + L))
    Ti = T
    return PIDController(Kp=Kp, Ti=Ti, Td=0.0, name="PI-Lambda")

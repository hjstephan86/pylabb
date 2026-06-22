"""
pylabb.simulation.time_domain
=============================
Zeitbereichssimulation von linearen und nichtlinearen Regelkreisen.

Enthält:
- ``ClosedLoopSimulator``  : Simuliert Führungs- und Störantwort eines
                             Regelkreises (Regler + Strecke) mit beliebigen
                             Eingangs- und Störsignalen.
- ``simulate_tf``          : Direkte Simulation einer ÜTF.
- ``simulate_ss``          : Direkte Simulation eines Zustandsraumsystems.
- ``SimulationResult``     : Ergebnisdatenklasse mit Kenngrößen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray
import scipy.signal as sp_sig
import scipy.integrate as sp_int

from pylabb.core.transfer_function import TransferFunction
from pylabb.core.state_space import StateSpace
from pylabb.core.signals import Signal


# ---------------------------------------------------------------------------
# Ergebnisdatenklasse
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Enthält das vollständige Ergebnis einer Zeitbereichssimulation.

    Attributes
    ----------
    t         : Zeitachse [s].
    y         : Ausgang(ssignal) der Strecke.
    u         : Stellgröße.
    e         : Regelfehler setpoint − y.
    setpoint  : Sollwertsignal.
    disturbance : Störsignal (soweit vorh.).
    name      : Bezeichner der Simulation.
    """
    t: NDArray
    y: NDArray
    u: NDArray
    e: NDArray
    setpoint: NDArray
    disturbance: NDArray = field(default_factory=lambda: np.array([0.0]))
    name: str = "Simulation"

    # ------------------------------------------------------------------
    # Kennwerte
    # ------------------------------------------------------------------

    def rise_time(self, low: float = 0.1, high: float = 0.9) -> float:
        """Anstiegszeit von *low* auf *high* des Endwerts (Sprungantwort)."""
        yf = self.y[-1]
        if abs(yf) < 1e-12:
            return float("nan")
        times_low = self.t[self.y >= low * yf]
        times_high = self.t[self.y >= high * yf]
        if not len(times_low) or not len(times_high):  # pragma: no cover
            return float("nan")
        return float(times_high[0] - times_low[0])

    def settling_time(self, band: float = 0.02) -> float:
        """Einschwingzeit: letzter Zeitpunkt, an dem y ± band*yf verlässt."""
        yf = self.y[-1]
        if abs(yf) < 1e-12:
            return float("nan")
        outside = np.abs(self.y - yf) > band * abs(yf)
        if not np.any(outside):
            return 0.0
        last_outside = np.where(outside)[0][-1]
        return float(self.t[last_outside])

    def overshoot_pct(self) -> float:
        """Überschwingweite in % bezogen auf den Endwert."""
        yf = self.y[-1]
        if abs(yf) < 1e-12:
            return float("nan")
        return float(100.0 * (np.max(self.y) - yf) / abs(yf))

    def steady_state_error(self) -> float:
        """Stationäre Regelabweichung e_∞ = w_∞ − y_∞."""
        return float(self.setpoint[-1] - self.y[-1])

    def iae(self) -> float:
        """Integral Absolute Error ∫|e|dt."""
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(_trapz(np.abs(self.e), self.t))

    def ise(self) -> float:
        """Integral Squared Error ∫e²dt."""
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(_trapz(self.e**2, self.t))

    def itae(self) -> float:
        """Integral Time-weighted Absolute Error ∫t|e|dt."""
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(_trapz(self.t * np.abs(self.e), self.t))

    def summary(self) -> dict:
        """Dictionary aller Kennwerte."""
        return {
            "Anstiegszeit [s]": self.rise_time(),
            "Einschwingzeit [s]": self.settling_time(),
            "Überschwingen [%]": self.overshoot_pct(),
            "Stationäre Abweichung": self.steady_state_error(),
            "IAE": self.iae(),
            "ISE": self.ise(),
            "ITAE": self.itae(),
        }

    def to_signal(self, channel: str = "y") -> Signal:
        """Gibt einen Kanal als ``Signal``-Objekt zurück."""
        data = {"y": self.y, "u": self.u, "e": self.e, "setpoint": self.setpoint}
        return Signal(self.t, data[channel], name=f"{self.name}.{channel}")


# ---------------------------------------------------------------------------
# Geschlossener Regelkreis
# ---------------------------------------------------------------------------

class ClosedLoopSimulator:
    """Simuliert einen linearen PID-Regelkreis im Zeitbereich.

    Der Regler C wird als diskrete ÜTF oder als
    ``DiscretePIDController``-Objekt übergeben. Die Strecke G wird
    entweder als ``TransferFunction``, ``StateSpace`` oder als
    Callable (nichtlinear!) akzeptiert.

    Beispiel::

        from pylabb.core.transfer_function import TransferFunction
        from pylabb.control.pid import PIDController, ziegler_nichols_step
        from pylabb.simulation.time_domain import ClosedLoopSimulator
        from pylabb.core.signals import step_signal

        G = TransferFunction([1], [1, 2, 1])
        pid = ziegler_nichols_step(K=1, T=1, L=0.1)
        C = pid.transfer_function()
        sim = ClosedLoopSimulator(G, C)
        result = sim.run(step_signal(t_end=10))
    """

    def __init__(
        self,
        plant: TransferFunction | StateSpace,
        controller: TransferFunction,
        disturbance_tf: Optional[TransferFunction] = None,
        name: str = "Regelkreis",
    ) -> None:
        self.plant = plant
        self.controller = controller
        self.disturbance_tf = disturbance_tf
        self.name = name

    def run(
        self,
        setpoint: Signal,
        disturbance: Optional[Signal] = None,
    ) -> SimulationResult:
        """Führt die Simulation aus.

        Parameters
        ----------
        setpoint    : Sollwertsignal.
        disturbance : Störsignal auf den Streckeneingang.

        Returns
        -------
        SimulationResult
        """
        t = setpoint.t
        w = setpoint.y

        d = disturbance.y if disturbance is not None else np.zeros_like(w)

        # Geschlossener Kreis: T = C*G / (1 + C*G)
        from pylabb.control.design import closed_loop
        G = self.plant if isinstance(self.plant, TransferFunction) \
            else self.plant.to_transfer_function()
        T = closed_loop(G, self.controller)

        _, y = T.lsim(w, t)
        e = w - y

        # Stellgröße: u = C * e
        _, u = self.controller.lsim(e, t)

        # Störantwort (additiv)
        if self.disturbance_tf is not None:
            _, d_out = self.disturbance_tf.lsim(d, t)
            y = y + d_out

        return SimulationResult(
            t=t, y=y, u=u, e=e, setpoint=w, disturbance=d, name=self.name
        )


# ---------------------------------------------------------------------------
# Direkte Simulationsfunktionen
# ---------------------------------------------------------------------------

def simulate_tf(
    G: TransferFunction,
    u: Signal,
) -> SimulationResult:
    """Simuliert die Antwort der ÜTF G auf das Eingangsignale u.

    Parameters
    ----------
    G : Übertragungsfunktion.
    u : Eingangssignal.

    Returns
    -------
    SimulationResult (nur y und u gefüllt).
    """
    t_out, y_out = G.lsim(u.y, u.t)
    e = np.zeros_like(y_out)
    return SimulationResult(
        t=t_out, y=y_out, u=u.y, e=e, setpoint=u.y, name=f"sim({G.name})"
    )


def simulate_ss(
    sys: StateSpace,
    u: Signal,
) -> SimulationResult:
    """Simuliert eine Zustandsraumdarstellung mit Eingangsignal u."""
    t_out, y_out = sys.lsim(u.y, u.t)
    e = np.zeros_like(y_out)
    return SimulationResult(
        t=t_out, y=y_out.flatten(), u=u.y, e=e, setpoint=u.y, name=f"sim({sys.name})"
    )


def step_response_analysis(
    G: TransferFunction,
    C: Optional[TransferFunction] = None,
    t_end: float = 20.0,
    dt: float = 0.01,
) -> SimulationResult:
    """Berechnet die Sprungantwort des offenen oder geschlossenen Kreises.

    Parameters
    ----------
    G    : Strecke.
    C    : Regler (None → offener Kreis).
    t_end, dt : Simulationszeit und Abtastzeit.
    """
    from pylabb.core.signals import step_signal
    from pylabb.control.design import closed_loop

    w = step_signal(t_end=t_end, dt=dt)
    if C is None:
        return simulate_tf(G, w)
    T = closed_loop(G, C)
    return simulate_tf(T, w)

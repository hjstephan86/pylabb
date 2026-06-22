"""
pylabb.core.signals
==================
Signalgenerierung und -verarbeitung für Analyse und Simulation.

Enthält die ``Signal``-Klasse sowie Fabrikfunktionen für gängige
Testeingangssignale (Sprung, Rampe, Sinus, Impuls, Rechteck, Chirp,
Rauschen).  Alle Signale arbeiten intern mit ``numpy``-Arrays und sind
damit direkt mit der Simulationsschicht und matplotlib kompatibel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Haupt-Datenklasse
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Enthält ein abgetastetes Signal mit Zeitachse und Metadaten.

    Attributes
    ----------
    t      : Zeitachse [s].
    y      : Signalamplituden.
    name   : Bezeichnung des Signals.
    unit_t : Einheit der Zeitachse (Standard: 's').
    unit_y : Einheit der Amplitude  (Standard: '').
    dt     : Abtastzeit [s]; wird automatisch aus *t* geschätzt.
    """

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    name: str = ""
    unit_t: str = "s"
    unit_y: str = ""
    dt: float = field(init=False)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.t.ndim != 1 or self.y.ndim != 1:
            raise ValueError("t und y müssen eindimensional sein.")
        if len(self.t) != len(self.y):
            raise ValueError("t und y müssen gleich lang sein.")
        self.dt = float(self.t[1] - self.t[0]) if len(self.t) > 1 else 1.0

    # ------------------------------------------------------------------
    # Arithmetik
    # ------------------------------------------------------------------

    def __add__(self, other: "Signal | float") -> "Signal":
        if isinstance(other, Signal):
            return Signal(self.t, self.y + other.y, name=f"({self.name}+{other.name})")
        return Signal(self.t, self.y + other, name=self.name)

    def __sub__(self, other: "Signal | float") -> "Signal":
        if isinstance(other, Signal):
            return Signal(self.t, self.y - other.y, name=f"({self.name}-{other.name})")
        return Signal(self.t, self.y - other, name=self.name)

    def __mul__(self, scalar: float) -> "Signal":
        return Signal(self.t, self.y * scalar, name=self.name)

    def __rmul__(self, scalar: float) -> "Signal":
        return self.__mul__(scalar)

    def __neg__(self) -> "Signal":
        return Signal(self.t, -self.y, name=f"-{self.name}")

    # ------------------------------------------------------------------
    # Eigenschaften
    # ------------------------------------------------------------------

    @property
    def duration(self) -> float:
        """Gesamtlänge des Signals [s]."""
        return float(self.t[-1] - self.t[0])

    @property
    def fs(self) -> float:
        """Abtastfrequenz [Hz]."""
        return 1.0 / self.dt if self.dt > 0 else float("inf")

    @property
    def n_samples(self) -> int:
        """Anzahl der Abtastwerte."""
        return len(self.t)

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def rms(self) -> float:
        """Effektivwert (Root Mean Square)."""
        return float(np.sqrt(np.mean(self.y ** 2)))

    def peak(self) -> float:
        """Spitzenwert (absoluter Maximalwert)."""
        return float(np.max(np.abs(self.y)))

    def energy(self) -> float:
        """Diskrete Signalenergie ∑ y² · dt."""
        return float(np.sum(self.y ** 2) * self.dt)

    def fft(self) -> tuple[NDArray, NDArray]:
        """Betragsspektrum via FFT.

        Returns
        -------
        freqs  : Frequenzachse [Hz] (nur positive Hälfte).
        magnitudes : Betrag des komplexen Spektrums.
        """
        n = len(self.y)
        freqs = np.fft.rfftfreq(n, d=self.dt)
        magnitudes = np.abs(np.fft.rfft(self.y)) / n
        return freqs, magnitudes

    def crop(self, t_start: float, t_end: float) -> "Signal":
        """Schneidet das Signal auf [t_start, t_end] zu."""
        mask = (self.t >= t_start) & (self.t <= t_end)
        return Signal(self.t[mask], self.y[mask], name=self.name,
                      unit_t=self.unit_t, unit_y=self.unit_y)

    def resample(self, dt_new: float) -> "Signal":
        """Resampelt das Signal auf eine neue Abtastzeit mittels linearer
        Interpolation."""
        t_new = np.arange(self.t[0], self.t[-1], dt_new)
        y_new = np.interp(t_new, self.t, self.y)
        return Signal(t_new, y_new, name=self.name,
                      unit_t=self.unit_t, unit_y=self.unit_y)

    def __repr__(self) -> str:
        return (
            f"Signal(name={self.name!r}, n={self.n_samples}, "
            f"dt={self.dt:.4g}s, duration={self.duration:.4g}s)"
        )


# ---------------------------------------------------------------------------
# Fabrikfunktionen
# ---------------------------------------------------------------------------

def _make_time(t_end: float, dt: float, t_start: float = 0.0) -> NDArray:
    """Erstellt eine gleichmäßige Zeitachse."""
    return np.arange(t_start, t_end + dt / 2, dt)


def step_signal(
    t_end: float = 5.0,
    dt: float = 0.01,
    t_step: float = 0.0,
    amplitude: float = 1.0,
    name: str = "Sprung",
) -> Signal:
    """Einheitssprungsignal (oder beliebige Amplitude).

    Parameters
    ----------
    t_end     : Endzeitpunkt [s].
    dt        : Abtastzeit [s].
    t_step    : Zeitpunkt des Sprungs [s].
    amplitude : Höhe des Sprungs.
    name      : Signalbezeichnung.
    """
    t = _make_time(t_end, dt)
    y = np.where(t >= t_step, amplitude, 0.0)
    return Signal(t, y, name=name, unit_y="")


def ramp_signal(
    t_end: float = 5.0,
    dt: float = 0.01,
    t_start: float = 0.0,
    slope: float = 1.0,
    name: str = "Rampe",
) -> Signal:
    """Rampenfunktion r(t) = slope·(t − t_start) für t ≥ t_start."""
    t = _make_time(t_end, dt)
    y = np.where(t >= t_start, slope * (t - t_start), 0.0)
    return Signal(t, y, name=name, unit_y="")


def sine_signal(
    t_end: float = 5.0,
    dt: float = 0.001,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    phase_deg: float = 0.0,
    offset: float = 0.0,
    name: str = "Sinus",
) -> Signal:
    """Sinusförmiges Signal.

    Parameters
    ----------
    frequency  : Frequenz [Hz].
    phase_deg  : Phasenwinkel [°].
    offset     : DC-Offset.
    """
    t = _make_time(t_end, dt)
    y = amplitude * np.sin(2 * np.pi * frequency * t + np.deg2rad(phase_deg)) + offset
    return Signal(t, y, name=name, unit_y="")


def impulse_signal(
    t_end: float = 5.0,
    dt: float = 0.01,
    t_impulse: float = 0.0,
    amplitude: float = 1.0,
    name: str = "Impuls",
) -> Signal:
    """Diskreter Dirac-Impuls (Amplitude = amplitude/dt → Einheitsimpuls)."""
    t = _make_time(t_end, dt)
    y = np.zeros_like(t)
    idx = np.argmin(np.abs(t - t_impulse))
    y[idx] = amplitude / dt
    return Signal(t, y, name=name, unit_y="")


def square_signal(
    t_end: float = 5.0,
    dt: float = 0.001,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    duty_cycle: float = 0.5,
    name: str = "Rechteck",
) -> Signal:
    """Rechtecksignal mit einstellbarem Tastverhältnis.

    Parameters
    ----------
    duty_cycle : Tastverhältnis [0..1].
    """
    from scipy.signal import square as sp_square
    t = _make_time(t_end, dt)
    y = amplitude * (sp_square(2 * np.pi * frequency * t, duty=duty_cycle) + 1) / 2
    return Signal(t, y, name=name, unit_y="")


def chirp_signal(
    t_end: float = 5.0,
    dt: float = 0.001,
    f0: float = 0.1,
    f1: float = 10.0,
    amplitude: float = 1.0,
    name: str = "Chirp",
) -> Signal:
    """Frequenzgangstestanregung: linearer Frequenz-Sweep von *f0* bis *f1*.

    Parameters
    ----------
    f0 : Startfrequenz [Hz].
    f1 : Endfrequenz [Hz].
    """
    from scipy.signal import chirp as sp_chirp
    t = _make_time(t_end, dt)
    y = amplitude * sp_chirp(t, f0=f0, f1=f1, t1=t_end, method="linear")
    return Signal(t, y, name=name, unit_y="")


def noise_signal(
    t_end: float = 5.0,
    dt: float = 0.001,
    std: float = 1.0,
    mean: float = 0.0,
    seed: Optional[int] = None,
    name: str = "Rauschen",
) -> Signal:
    """Normalverteiltes weißes Rauschen.

    Parameters
    ----------
    std  : Standardabweichung (RMS-Amplitude).
    mean : Mittelwert.
    seed : Zufallssaat für Reproduzierbarkeit.
    """
    rng = np.random.default_rng(seed)
    t = _make_time(t_end, dt)
    y = rng.normal(loc=mean, scale=std, size=len(t))
    return Signal(t, y, name=name, unit_y="")


# Alias für Rückwärtskompatibilität
white_noise = noise_signal

"""
PyLab – Umfassendes Framework für mathematische Berechnungen,
Regelungstechnik, Automatisierung und Visualisierung mit
MicroPython-Codegenerierung.

Struktur
--------
pylabb.core          – Signale, Übertragungsfunktionen, Zustandsraum,
                      Polynomoperationen, Math-Utilities
pylabb.control       – PID-Regler (kontinuierlich / diskret), Einstellregeln,
                      Stabilitätsanalyse, Lead/Lag, Loopshaping
pylabb.simulation    – Zeitbereichssimulation geschlossener Regelkreise
pylabb.visualization – Bode, Nyquist, Wurzelortskurve, Sprungantwort,
                      Spektrum, Pol-Nullstellen-Diagramm
pylabb.codegen       – MicroPython-Codegenerierung für Embedded Devices
pylabb.gui           – PyQt6-basierte grafische Benutzeroberfläche

Schnellstart
------------
>>> import pylab
>>> from pylabb.core.transfer_function import TransferFunction
>>> from pylabb.control.pid import PIDController, ziegler_nichols_step
>>> from pylabb.visualization import plot_bode, plot_step_response
>>>
>>> G = TransferFunction([1], [1, 1.4, 1])   # PT2
>>> pid = ziegler_nichols_step(K=1, T=1, L=0.1)
>>> plot_bode(G, pid.transfer_function())
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pylabb")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "2.1.0"

__author__ = "Stephan Epp"
__all__ = [
    "core",
    "control",
    "simulation",
    "visualization",
    "codegen",
    "gui",
]

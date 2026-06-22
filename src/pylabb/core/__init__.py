"""
pylabb.core – Grundlegende Mathematik, Signalverarbeitung und Systemdarstellungen.
"""

from .math_utils import (
    poly_eval,
    poly_multiply,
    poly_add,
    roots_of_polynomial,
    partial_fraction,
    bilinear_transform,
    zoh_discretize,
    routh_array,
)
from .signals import (
    Signal,
    step_signal,
    ramp_signal,
    sine_signal,
    impulse_signal,
    square_signal,
    chirp_signal,
    noise_signal,
    white_noise,
)
from .transfer_function import TransferFunction
from .state_space import StateSpace

__all__ = [
    # math_utils
    "poly_eval",
    "poly_multiply",
    "poly_add",
    "roots_of_polynomial",
    "partial_fraction",
    "bilinear_transform",
    "zoh_discretize",
    "routh_array",
    # signals
    "Signal",
    "step_signal",
    "ramp_signal",
    "sine_signal",
    "impulse_signal",
    "square_signal",
    "chirp_signal",
    "noise_signal",
    "white_noise",
    # systems
    "TransferFunction",
    "StateSpace",
]

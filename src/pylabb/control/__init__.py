"""
pylabb.control – Regelungstechnik-Bibliothek

Exportiert:
  PIDController, DiscretePIDController  – PID-Regler (kont./diskret)
  Einstellregeln: ziegler_nichols_step/oscillation, cohen_coon, lambda_tuning
  Stabilitätsanalyse: analyze, bode_margins, root_locus_data, critical_gain
  Reglerentwurf: lead_compensator, lag_compensator, notch_filter,
                 closed_loop, sensitivity, pade_approximation,
                 first_order_plant, second_order_plant
  Verifikation: verify_loops, loop_to_graph, VerificationResult
"""

from .pid import (
    PIDController,
    DiscretePIDController,
    ziegler_nichols_step,
    ziegler_nichols_oscillation,
    cohen_coon,
    lambda_tuning,
)
from .stability import (
    StabilityInfo,
    analyze,
    bode_margins,
    pole_zero_info,
    root_locus_data,
    critical_gain,
)
from .design import (
    lead_compensator,
    lag_compensator,
    lead_lag_compensator,
    notch_filter,
    bandpass_filter,
    lowpass_filter,
    highpass_filter,
    closed_loop,
    sensitivity,
    complementary_sensitivity,
    control_sensitivity,
    series,
    parallel,
    pade_approximation,
    first_order_plant,
    second_order_plant,
    integrating_plant,
)
from .verification import (
    VerificationResult,
    verify_loops,
    loop_to_graph,
)
from .bio_classify import (
    BioClass,
    BioExtension,
    BioClassResult,
    classify_loop,
    get_extensions,
    get_reference_topology,
)

__all__ = [
    # PID
    "PIDController",
    "DiscretePIDController",
    "ziegler_nichols_step",
    "ziegler_nichols_oscillation",
    "cohen_coon",
    "lambda_tuning",
    # Stabilität
    "StabilityInfo",
    "analyze",
    "bode_margins",
    "pole_zero_info",
    "root_locus_data",
    "critical_gain",
    # Entwurf
    "lead_compensator",
    "lag_compensator",
    "lead_lag_compensator",
    "notch_filter",
    "bandpass_filter",
    "lowpass_filter",
    "highpass_filter",
    "closed_loop",
    "sensitivity",
    "complementary_sensitivity",
    "control_sensitivity",
    "series",
    "parallel",
    "pade_approximation",
    "first_order_plant",
    "second_order_plant",
    "integrating_plant",
    # Verifikation
    "VerificationResult",
    "verify_loops",
    "loop_to_graph",
    # Biologische Äquivalenzklassen
    "BioClass",
    "BioExtension",
    "BioClassResult",
    "classify_loop",
    "get_extensions",
    "get_reference_topology",
]

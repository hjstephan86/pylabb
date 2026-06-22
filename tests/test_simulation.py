"""
Tests für pylabb.simulation.time_domain
"""
import numpy as np
import pytest
from pylabb.simulation.time_domain import (
    SimulationResult,
    simulate_tf,
    step_response_analysis,
)
from pylabb.core.transfer_function import TransferFunction
from pylabb.core.signals import step_signal


class TestSimulationResult:
    def _result(self) -> SimulationResult:
        G = TransferFunction([1], [1, 1])
        t = np.linspace(0, 10, 1000)
        _, y = G.step_response(t_end=10)
        t_g, y_g = G.step_response(t_end=10)
        u = np.ones_like(t_g)
        e = u - y_g
        return SimulationResult(t=t_g, y=y_g, u=u, e=e, setpoint=u)

    def test_steady_state_error_pt1(self):
        res = self._result()
        sse = res.steady_state_error()
        assert abs(sse) < 0.05  # PT1 bei Einheitssprung → sse ≈ 0

    def test_overshoot_no_overshoot(self):
        res = self._result()
        os = res.overshoot_pct()
        assert os < 1.0  # PT1 hat kein Überschwingen

    def test_iae_positive(self):
        res = self._result()
        assert res.iae() > 0

    def test_summary_keys(self):
        res = self._result()
        s = res.summary()
        assert "IAE" in s
        assert "ISE" in s
        assert "ITAE" in s

    def test_to_signal(self):
        res = self._result()
        sig = res.to_signal("y")
        assert sig.name.endswith("y")
        assert len(sig.y) > 0


class TestSimulateTF:
    def test_output_shape(self):
        G = TransferFunction([1], [1, 1])
        u = step_signal(t_end=5.0, dt=0.01)
        res = simulate_tf(G, u)
        assert len(res.t) == len(res.y)
        assert len(res.y) > 0

    def test_step_response_final_value(self):
        G = TransferFunction([1], [1, 1])
        u = step_signal(t_end=20.0, dt=0.01)
        res = simulate_tf(G, u)
        assert abs(res.y[-1] - 1.0) < 0.05


class TestStepResponseAnalysis:
    def test_open_loop(self):
        G = TransferFunction([1], [1, 1])
        res = step_response_analysis(G, t_end=10.0)
        assert len(res.y) > 0

    def test_closed_loop(self):
        from pylabb.control.pid import PIDController
        G = TransferFunction([1], [1, 2, 1])
        C = PIDController(Kp=2.0, Ti=1.0, Td=0.1).transfer_function()
        res = step_response_analysis(G, C=C, t_end=15.0)
        assert len(res.y) > 0

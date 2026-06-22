"""
Erweiterte Tests für pylabb.codegen.micropython – 100 % Pfadabdeckung
"""
import numpy as np
import pytest

from pylabb.codegen.micropython import (
    CodegenConfig,
    _CodeBuilder,
    gen_pid,
    gen_digital_filter,
    gen_state_feedback,
    gen_control_loop,
    gen_project,
    _mat_to_list,
    _fmt,
)
from pylabb.control.pid import PIDController, DiscretePIDController
from pylabb.core.transfer_function import TransferFunction
from pylabb.core.state_space import StateSpace


# ---------------------------------------------------------------------------
# _CodeBuilder
# ---------------------------------------------------------------------------

class TestCodeBuilder:
    def test_line_and_build(self):
        cb = _CodeBuilder()
        cb.line("x = 1")
        assert "x = 1" in cb.build()

    def test_blank(self):
        cb = _CodeBuilder()
        cb.blank()
        assert cb.build() == ""

    def test_comment(self):
        cb = _CodeBuilder()
        cb.comment("hello")
        assert "# hello" in cb.build()

    def test_indent_dedent(self):
        cb = _CodeBuilder(indent="    ")
        cb.indent()
        cb.line("indented")
        cb.dedent()
        cb.line("not indented")
        src = cb.build()
        assert "    indented" in src
        assert src.count("    indented") == 1
        assert "not indented" in src

    def test_dedent_not_below_zero(self):
        cb = _CodeBuilder()
        cb.dedent()  # must not raise
        cb.line("ok")
        assert "ok" in cb.build()

    def test_raw(self):
        cb = _CodeBuilder()
        cb.raw("line1\nline2")
        src = cb.build()
        assert "line1" in src
        assert "line2" in src


# ---------------------------------------------------------------------------
# _mat_to_list / _fmt
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_mat_to_list_2d(self):
        M = np.array([[1, 2], [3, 4]], dtype=float)
        s = _mat_to_list(M)
        assert "[[" in s

    def test_mat_to_list_1d(self):
        v = np.array([1.0, 2.0, 3.0])
        s = _mat_to_list(v)
        assert s.startswith("[")
        assert "[[" not in s

    def test_fmt_inf(self):
        assert _fmt(float("inf")) == "1e18"
        assert _fmt(float("-inf")) == "-1e18"

    def test_fmt_normal(self):
        s = _fmt(3.14)
        assert "3.14" in s


# ---------------------------------------------------------------------------
# gen_pid – PIDController path
# ---------------------------------------------------------------------------

class TestGenPidContinuous:
    def _make_pid(self, Kp=1.0, Ti=1.0, Td=0.0):
        return PIDController(Kp=Kp, Ti=Ti, Td=Td)

    def test_basic_pid_code_contains_class(self):
        pid = self._make_pid()
        code = gen_pid(pid)
        assert "class PIDController" in code

    def test_ti_infinity_no_integral(self):
        """Ti = inf → I-action disabled."""
        pid = PIDController(Kp=2.0, Ti=float("inf"), Td=0.0)
        code = gen_pid(pid)
        assert "I = 0.0" in code

    def test_td_positive_derivative_active(self):
        """Td > 0 → D-filter code emitted."""
        pid = PIDController(Kp=1.0, Ti=1.0, Td=0.1)
        code = gen_pid(pid)
        assert "alpha" in code

    def test_no_comments(self):
        cfg = CodegenConfig(add_comments=False)
        pid = self._make_pid()
        code = gen_pid(pid, cfg=cfg)
        # should still produce class definition
        assert "class PIDController" in code

    def test_custom_class_name(self):
        pid = self._make_pid()
        code = gen_pid(pid, class_name="MyPID")
        assert "class MyPID" in code


# ---------------------------------------------------------------------------
# gen_pid – DiscretePIDController path
# ---------------------------------------------------------------------------

class TestGenPidDiscrete:
    def test_discrete_pid_code(self):
        dpid = DiscretePIDController(Kp=1.5, Ti=2.0, Td=0.05, dt=0.01)
        code = gen_pid(dpid, class_name="DPID")
        assert "class DPID" in code
        # dt should be taken from the discrete controller
        assert "0.01" in code

    def test_discrete_pid_no_integral(self):
        dpid = DiscretePIDController(Kp=1.0, Ti=float("inf"), Td=0.0, dt=0.01)
        code = gen_pid(dpid)
        assert "I = 0.0" in code

    def test_discrete_pid_with_limits(self):
        dpid = DiscretePIDController(Kp=1.0, Ti=1.0, Td=0.0, dt=0.01,
                                     u_min=-5.0, u_max=5.0)
        code = gen_pid(dpid)
        assert "5" in code


# ---------------------------------------------------------------------------
# gen_digital_filter
# ---------------------------------------------------------------------------

class TestGenDigitalFilter:
    def test_generates_class(self):
        G = TransferFunction([1], [1, 1])
        code = gen_digital_filter(G, dt=0.01)
        assert "class DigitalFilter" in code

    def test_custom_class_name(self):
        G = TransferFunction([1], [1, 1])
        code = gen_digital_filter(G, dt=0.01, class_name="LP")
        assert "class LP" in code

    def test_euler_method(self):
        """2nd-order Euler discretization → order=2 → range(1) has one iteration → covers loop body line."""
        G = TransferFunction([1], [1, 1.5, 0.5])  # 2nd-order system
        code = gen_digital_filter(G, dt=0.01, method="euler")
        assert "class DigitalFilter" in code


# ---------------------------------------------------------------------------
# gen_state_feedback
# ---------------------------------------------------------------------------

class TestGenStateFeedback:
    def _discrete_ss(self):
        A = np.array([[-1.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])
        ss = StateSpace(A, B, C, D)
        ss.dt = 0.01
        return ss

    def test_without_observer(self):
        ss = self._discrete_ss()
        K = np.array([[2.0]])
        code = gen_state_feedback(ss, K)
        assert "class StateFeedbackController" in code
        assert "def update" in code

    def test_with_observer(self):
        ss = self._discrete_ss()
        K = np.array([[2.0]])
        L = np.array([[0.5]])
        code = gen_state_feedback(ss, K, L=L)
        assert "self.L" in code
        assert "innov" in code

    def test_raises_if_no_dt(self):
        A = np.array([[-1.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])
        ss = StateSpace(A, B, C, D)
        K = np.array([[2.0]])
        with pytest.raises(ValueError, match="zeitdiskret"):
            gen_state_feedback(ss, K)

    def test_matrix_helpers_included(self):
        ss = self._discrete_ss()
        K = np.array([[2.0]])
        code = gen_state_feedback(ss, K)
        assert "_matvec" in code

    def test_custom_class_name(self):
        ss = self._discrete_ss()
        K = np.array([[2.0]])
        code = gen_state_feedback(ss, K, class_name="SFC")
        assert "class SFC" in code


# ---------------------------------------------------------------------------
# gen_control_loop
# ---------------------------------------------------------------------------

class TestGenControlLoop:
    def _pid_code(self):
        pid = PIDController(Kp=1.0, Ti=1.0, Td=0.0)
        return gen_pid(pid)

    def test_basic(self):
        code = gen_control_loop(self._pid_code())
        assert "while True" in code
        assert "PIDController" in code

    def test_with_uart(self):
        cfg = CodegenConfig(include_uart=True)
        code = gen_control_loop(self._pid_code(), cfg=cfg)
        assert "UART" in code

    def test_with_watchdog(self):
        cfg = CodegenConfig(include_watchdog=True)
        code = gen_control_loop(self._pid_code(), cfg=cfg)
        assert "WDT" in code

    def test_custom_dt_and_funcs(self):
        code = gen_control_loop(
            self._pid_code(),
            dt_ms=20,
            sensor_func="my_sensor",
            actuator_func="set_pwm",
            setpoint_func="get_ref",
        )
        assert "dt_ms = 20" in code
        assert "my_sensor" in code
        assert "set_pwm" in code
        assert "get_ref" in code


# ---------------------------------------------------------------------------
# gen_project
# ---------------------------------------------------------------------------

class TestGenProject:
    def test_returns_dict_with_keys(self):
        pid = PIDController(Kp=1.0, Ti=2.0, Td=0.0)
        result = gen_project(pid, dt_ms=10)
        assert set(result.keys()) == {"main.py", "pid.py", "README.md"}

    def test_main_has_import(self):
        pid = PIDController(Kp=1.0, Ti=2.0, Td=0.0)
        result = gen_project(pid, dt_ms=10)
        assert "from pid import PIDController" in result["main.py"]

    def test_readme_has_params(self):
        pid = PIDController(Kp=2.5, Ti=3.0, Td=0.0)
        result = gen_project(pid, dt_ms=10)
        assert "2.5" in result["README.md"]

    def test_no_comments_cfg(self):
        cfg = CodegenConfig(add_comments=False)
        pid = PIDController(Kp=1.0, Ti=1.0, Td=0.0)
        result = gen_project(pid, cfg=cfg)
        assert "class PIDController" in result["pid.py"]

"""
Tests für pylabb.codegen.micropython
"""
import pytest
from pylabb.codegen.micropython import (
    CodegenConfig,
    gen_pid,
    gen_digital_filter,
    gen_control_loop,
    gen_project,
)
from pylabb.control.pid import PIDController
from pylabb.core.transfer_function import TransferFunction


class TestGenPID:
    def test_generates_class(self):
        pid = PIDController(Kp=1.5, Ti=2.0, Td=0.1)
        code = gen_pid(pid, class_name="MyPID")
        assert "class MyPID:" in code
        assert "def update(self" in code
        assert "def reset(self)" in code
        assert "1.5" in code  # Kp

    def test_includes_comments(self):
        pid = PIDController(Kp=1.0)
        cfg = CodegenConfig(add_comments=True)
        code = gen_pid(pid, cfg=cfg)
        assert "#" in code

    def test_no_comments(self):
        pid = PIDController(Kp=1.0)
        cfg = CodegenConfig(add_comments=False)
        code = gen_pid(pid, cfg=cfg)
        # Noch immer gültig, aber weniger Kommentare
        assert "class PIDController:" in code

    def test_saturation_code(self):
        pid = PIDController(Kp=1.0)
        code = gen_pid(pid)
        assert "u_max" in code
        assert "u_min" in code

    def test_anti_windup_present(self):
        pid = PIDController(Kp=1.0, Ti=1.0, Td=0.0)
        code = gen_pid(pid)
        assert "Anti-Windup" in code or "anti" in code.lower() or "windup" in code.lower()


class TestGenDigitalFilter:
    def test_generates_class(self):
        G = TransferFunction([1], [1, 1])
        code = gen_digital_filter(G, dt=0.01, class_name="MyFilter")
        assert "class MyFilter:" in code
        assert "def update(self" in code
        assert "self.b" in code
        assert "self.a" in code

    def test_discrete_coefficients_in_code(self):
        G = TransferFunction([1], [1, 1])
        code = gen_digital_filter(G, dt=0.01)
        # Diskretisierte Koeffizienten sollten im Code stehen
        assert "self.b = [" in code


class TestGenControlLoop:
    def test_generates_main_loop(self):
        pid = PIDController(Kp=1.0)
        pid_code = gen_pid(pid)
        code = gen_control_loop(
            pid_code,
            controller_class="PIDController",
            dt_ms=10,
        )
        assert "while True:" in code
        assert "controller" in code
        assert "time.sleep_ms" in code

    def test_uart_option(self):
        pid = PIDController(Kp=1.0)
        pid_code = gen_pid(pid)
        cfg = CodegenConfig(include_uart=True)
        code = gen_control_loop(pid_code, cfg=cfg)
        assert "uart" in code.lower() or "UART" in code

    def test_watchdog_option(self):
        pid = PIDController(Kp=1.0)
        pid_code = gen_pid(pid)
        cfg = CodegenConfig(include_watchdog=True)
        code = gen_control_loop(pid_code, cfg=cfg)
        assert "wdt" in code


class TestGenProject:
    def test_returns_expected_files(self):
        pid = PIDController(Kp=1.0, Ti=1.0, Td=0.05)
        files = gen_project(pid)
        assert "main.py" in files
        assert "pid.py" in files
        assert "README.md" in files

    def test_main_imports_pid(self):
        pid = PIDController(Kp=1.0)
        files = gen_project(pid)
        assert "pid" in files["main.py"].lower() or "import" in files["main.py"]

    def test_readme_contains_params(self):
        pid = PIDController(Kp=2.5, Ti=1.0, Td=0.05)
        files = gen_project(pid)
        assert "2.5" in files["README.md"]

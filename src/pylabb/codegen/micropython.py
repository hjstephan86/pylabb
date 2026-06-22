"""
pylabb.codegen.micropython
==========================
Codegenerator für MicroPython-Targets.

Generiert vollständigen, lauffähigen MicroPython-Code für:

* PID-Regler         (``gen_pid``)
* Digitaler Filter / diskrete ÜTF  (``gen_digital_filter``)
* Zustandsraumregler  (``gen_state_feedback``)
* Vollständiger Regelkreis mit Hauptschleife (``gen_control_loop``)

Der generierte Code ist kompatibel mit MicroPython ≥ 1.20 (ESP32,
RP2040, STM32) und benötigt keine externen Bibliotheken – nur den
MicroPython-Standard und optional ``machine``.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np

from pylabb.control.pid import PIDController, DiscretePIDController
from pylabb.core.transfer_function import TransferFunction
from pylabb.core.state_space import StateSpace


# ---------------------------------------------------------------------------
# Konfigurations-Datenklasse
# ---------------------------------------------------------------------------

@dataclass
class CodegenConfig:
    """Konfigurationsoptionen für die MicroPython-Codegenerierung.

    Attributes
    ----------
    float_type     : 'float' (Standard) or 'double' (nicht alle MCUs).
    indent         : Einrück-String.
    add_comments   : Docstrings und Kommentare einfügen.
    target         : Zielplattform-Hinweis für den Header-Kommentar.
    tick_source    : 'time.ticks_ms' | 'machine.RTC' | 'custom'.
    include_uart   : Fügt einfache UART-Ausgabe ein.
    include_watchdog : Fügt Watchdog-Reset ein.
    """
    float_type: str = "float"
    indent: str = "    "
    add_comments: bool = True
    target: str = "ESP32 / RP2040 / STM32"
    tick_source: Literal["time.ticks_ms", "machine.RTC", "custom"] = "time.ticks_ms"
    include_uart: bool = False
    include_watchdog: bool = False


# ---------------------------------------------------------------------------
# Interner Hilfs-Code-Builder
# ---------------------------------------------------------------------------

class _CodeBuilder:
    """Einfacher String-Builder für eingerückten Code."""

    def __init__(self, indent: str = "    ") -> None:
        self._lines: list[str] = []
        self._level: int = 0
        self._indent = indent

    def line(self, text: str = "") -> "_CodeBuilder":
        if text:
            self._lines.append(self._indent * self._level + text)
        else:
            self._lines.append("")
        return self

    def blank(self) -> "_CodeBuilder":
        return self.line()

    def comment(self, text: str) -> "_CodeBuilder":
        return self.line(f"# {text}")

    def indent(self) -> "_CodeBuilder":
        self._level += 1
        return self

    def dedent(self) -> "_CodeBuilder":
        self._level = max(0, self._level - 1)
        return self

    def build(self) -> str:
        return "\n".join(self._lines)

    def raw(self, text: str) -> "_CodeBuilder":
        """Fügt mehrzeiligen Text auf aktueller Einrückebene ein."""
        for l in text.splitlines():
            self.line(l)
        return self


# ---------------------------------------------------------------------------
# PID-Generator
# ---------------------------------------------------------------------------

def gen_pid(
    controller: PIDController | DiscretePIDController,
    class_name: str = "PIDController",
    cfg: Optional[CodegenConfig] = None,
) -> str:
    """Generiert eine MicroPython-PID-Klasse.

    Parameters
    ----------
    controller : PIDController oder DiscretePIDController.
    class_name : Name der generierten Klasse.
    cfg        : Codegen-Konfiguration.

    Returns
    -------
    str : Vollständiger MicroPython-Quellcode.
    """
    cfg = cfg or CodegenConfig()

    # Extrahiere Parameter
    if isinstance(controller, DiscretePIDController):
        Kp = controller.Kp
        Ti = controller.Ti
        Td = controller.Td
        dt = controller.dt
        u_min = controller.u_min
        u_max = controller.u_max
        N = controller.N
    else:
        Kp = controller.Kp
        Ti = controller.Ti
        Td = controller.Td
        dt = 0.01  # Standard-Abtastzeit
        u_min = -1e9
        u_max = 1e9
        N = controller.N

    b = _CodeBuilder(cfg.indent)
    _write_file_header(b, cfg, "PID-Regler")

    b.line("import time")
    b.blank()
    b.blank()

    if cfg.add_comments:
        b.comment(f"Generiert von pylabb.codegen für {cfg.target}")
        b.comment(f"Kp={Kp:.6g}  Ti={Ti:.6g}  Td={Td:.6g}  dt={dt:.6g}s")
        b.blank()

    b.line(f"class {class_name}:")
    b.indent()

    if cfg.add_comments:
        b.comment("Zeitdiskreter PID-Regler (Positionsalgorithmus mit D-Filter und Anti-Windup)")
        b.blank()

    # __init__
    b.line("def __init__(self):")
    b.indent()
    b.comment("Regelparameter")
    b.line(f"self.Kp = {Kp:.8g}")
    b.line(f"self.Ti = {Ti:.8g}" if Ti != float("inf") else "self.Ti = None  # kein I-Anteil")
    b.line(f"self.Td = {Td:.8g}")
    b.line(f"self.dt = {dt:.8g}  # Abtastzeit [s]")
    b.line(f"self.N  = {N:.8g}  # D-Filterkoeffizient")
    b.line(f"self.u_min = {_fmt(u_min)}")
    b.line(f"self.u_max = {_fmt(u_max)}")
    b.comment("Interne Zustände")
    b.line("self._integral = 0.0")
    b.line("self._prev_error = 0.0")
    b.line("self._prev_deriv = 0.0")
    b.dedent()
    b.blank()

    # reset
    b.line("def reset(self):")
    b.indent()
    b.comment("Setzt alle internen Zustände zurück")
    b.line("self._integral = 0.0")
    b.line("self._prev_error = 0.0")
    b.line("self._prev_deriv = 0.0")
    b.dedent()
    b.blank()

    # update
    b.line("def update(self, setpoint, measurement):")
    b.indent()
    if cfg.add_comments:
        b.comment("Berechnet Stellgröße u für einen Regelschritt")
        b.comment("setpoint    : Sollwert")
        b.comment("measurement : Istwert")
        b.comment("Rückgabe    : Stellgröße (geclippt)")
        b.blank()
    b.line("error = setpoint - measurement")
    b.blank()
    b.comment("Proportionalanteil")
    b.line("P = self.Kp * error")
    b.blank()
    b.comment("Integralanteil")
    if Ti != float("inf") and Ti > 0:
        b.line("self._integral += error * self.dt")
        b.line("I = self.Kp / self.Ti * self._integral")
    else:
        b.line("I = 0.0")
    b.blank()
    b.comment("Differentialanteil (Tiefpassfilter)")
    if Td > 0:
        b.line("alpha = self.Td / (self.Td + self.N * self.dt)")
        b.line("deriv = alpha * self._prev_deriv + (1.0 - alpha) * (error - self._prev_error) / self.dt")
        b.line("D = self.Kp * self.Td * deriv")
        b.line("self._prev_deriv = deriv")
    else:
        b.line("D = 0.0")
    b.blank()
    b.line("u_raw = P + I + D")
    b.blank()
    b.comment("Stellgrößenbegrenzung")
    b.line("if u_raw > self.u_max:")
    b.indent()
    b.line("u = self.u_max")
    b.dedent()
    b.line("elif u_raw < self.u_min:")
    b.indent()
    b.line("u = self.u_min")
    b.dedent()
    b.line("else:")
    b.indent()
    b.line("u = u_raw")
    b.dedent()
    b.blank()
    b.comment("Anti-Windup (Rückverfolgung)")
    if Ti != float("inf") and Ti > 0:
        b.line("if self.Ti > 0 and self.Kp != 0:")
        b.indent()
        b.line("self._integral += (u - u_raw) / (self.Kp / self.Ti) * self.dt")
        b.dedent()
    b.blank()
    b.line("self._prev_error = error")
    b.line("return u")
    b.dedent()
    b.dedent()

    b.blank()
    b.blank()
    _write_main_loop_pid(b, class_name, dt, cfg)

    return b.build()


# ---------------------------------------------------------------------------
# Digitaler Filter (diskrete ÜTF)
# ---------------------------------------------------------------------------

def gen_digital_filter(
    G: TransferFunction,
    dt: float,
    method: str = "tustin",
    class_name: str = "DigitalFilter",
    cfg: Optional[CodegenConfig] = None,
) -> str:
    """Generiert einen MicroPython-Direktform-II-Filter (diskrete ÜTF).

    Die kontinuierliche ÜTF wird zuerst diskretisiert, dann als
    Direct-Form-II-Transponiert-Struktur implementiert.

    Parameters
    ----------
    G          : Kontinuierliche Übertragungsfunktion.
    dt         : Abtastzeit [s].
    method     : Diskretisierungsmethode ('tustin', 'zoh', 'euler').
    class_name : Name der generierten Klasse.
    cfg        : CodegenConfig.
    """
    cfg = cfg or CodegenConfig()
    G_d = G.discretize(dt, method=method)

    b_coeffs = G_d.num.real.tolist()
    a_coeffs = G_d.den.real.tolist()

    # Normierung auf a[0] = 1
    a0 = a_coeffs[0]
    b_coeffs = [x / a0 for x in b_coeffs]
    a_coeffs = [x / a0 for x in a_coeffs]

    order = len(a_coeffs) - 1

    code = _CodeBuilder(cfg.indent)
    _write_file_header(code, cfg, f"Digitaler Filter (DF-II) – {G.name}")

    code.blank()
    if cfg.add_comments:
        code.comment(f"Diskrete ÜTF aus: {G!r}")
        code.comment(f"Diskretisierung: {method}, dt={dt:.6g}s")
        code.comment(f"b = {[round(x,8) for x in b_coeffs]}")
        code.comment(f"a = {[round(x,8) for x in a_coeffs]}")
        code.blank()

    code.line(f"class {class_name}:")
    code.indent()
    code.line("def __init__(self):")
    code.indent()
    b_str = "[" + ", ".join(f"{v:.8g}" for v in b_coeffs) + "]"
    a_str = "[" + ", ".join(f"{v:.8g}" for v in a_coeffs) + "]"
    code.line(f"self.b = {b_str}  # Zählerkoeffizienten (diskret)")
    code.line(f"self.a = {a_str}  # Nennerkoeffizienten (diskret)")
    code.line(f"self._w = [0.0] * {order}  # Verzögerungsspeicher")
    code.dedent()
    code.blank()

    code.line("def reset(self):")
    code.indent()
    code.line(f"self._w = [0.0] * {order}")
    code.dedent()
    code.blank()

    code.line("def update(self, x):")
    code.indent()
    if cfg.add_comments:
        code.comment("Direct-Form-II transponiert")
        code.comment("x: Eingangswert, Rückgabe: Ausgangswert")
    code.line("b = self.b")
    code.line("a = self.a")
    code.line("w = self._w")
    code.line("v = x")
    for i in range(1, order + 1):
        code.line(f"v -= a[{i}] * w[{i-1}]" if i <= order else "")
    # Direct Form II transposed implementation
    code.line("# DF-II transponierte Berechnung")
    code.line("w_in = x")
    for i in range(1, min(order + 1, len(a_coeffs))):
        code.line(f"w_in -= a[{i}] * w[{i-1}]" if i - 1 < order else "")
    code.line(f"y = b[0] * w_in" + (" + w[0]" if order > 0 else ""))
    if order > 0:
        code.line("# Zustands-Update")
        for i in range(order - 1):
            code.line(f"w[{i}] = b[{i+1}] * w_in - a[{i+1}] * w_in + w[{i+1}]"
                      if i + 1 < order else f"w[{i}] = b[{i+1}] * w_in")
        code.line(f"w[{order-1}] = b[{order}] * w_in" if order < len(b_coeffs)
                  else f"w[{order-1}] = 0.0")
    code.line("return y")
    code.dedent()
    code.dedent()

    return code.build()


# ---------------------------------------------------------------------------
# Zustandsrückführung / Zustandsregler
# ---------------------------------------------------------------------------

def gen_state_feedback(
    sys: StateSpace,
    K: np.ndarray,
    L: Optional[np.ndarray] = None,
    class_name: str = "StateFeedbackController",
    cfg: Optional[CodegenConfig] = None,
) -> str:
    """Generiert einen Zustandsrückführungsregler (optional mit Beobachter).

    u(k) = −K·x̂(k)   (mit Beobachter: x̂ wird aus u,y geschätzt)

    Parameters
    ----------
    sys        : Zeitdiskretes Zustandsraumsystem (muss bereits diskretisiert sein).
    K          : Rückführungsmatrix (m×n).
    L          : Beobachter-Rückführungsmatrix (n×p); None → direktes x.
    class_name : Klassenname.
    cfg        : CodegenConfig.
    """
    cfg = cfg or CodegenConfig()
    if sys.dt is None:
        raise ValueError("Zustandssystem muss bereits zeitdiskret sein (sys.dt muss gesetzt sein).")

    n = sys.order
    m = sys.n_inputs
    p = sys.n_outputs

    b = _CodeBuilder(cfg.indent)
    _write_file_header(b, cfg, "Zustandsrückführungsregler")

    b.blank()
    if cfg.add_comments:
        b.comment(f"Systemordnung n={n}, Eingänge m={m}, Ausgänge p={p}")
        b.comment(f"Abtastzeit dt={sys.dt:.6g}s")
        b.blank()

    b.line(f"class {class_name}:")
    b.indent()
    b.line("def __init__(self):")
    b.indent()
    b.comment("Systemmatrizen (diskretisiert)")
    b.line(f"self.A = {_mat_to_list(sys.A)}")
    b.line(f"self.B = {_mat_to_list(sys.B)}")
    b.line(f"self.C = {_mat_to_list(sys.C)}")
    b.line(f"self.D = {_mat_to_list(sys.D)}")
    b.line(f"self.K = {_mat_to_list(K)}")
    if L is not None:
        b.line(f"self.L = {_mat_to_list(L)}")
    b.line(f"self.dt = {sys.dt:.8g}")
    b.comment("Zustandsvektor")
    b.line(f"self.x = [0.0] * {n}")
    b.dedent()
    b.blank()

    b.line("def reset(self):")
    b.indent()
    b.line(f"self.x = [0.0] * {n}")
    b.dedent()
    b.blank()

    b.line("def update(self, y, r=0.0):")
    b.indent()
    if cfg.add_comments:
        b.comment("y : gemessener Ausgang (Skalar für SISO)")
        b.comment("r : Sollwert (Referenzsignal)")
    b.line("x = self.x")
    b.line("A = self.A")
    b.line("B = self.B")
    b.line("C = self.C")
    b.line("K = self.K")
    b.blank()
    if L is not None:
        b.comment("Beobachterkorrektur")
        b.line("L = self.L")
        b.line("y_hat = _matvec(C, x)")
        b.line("innov = [y - y_hat[i] for i in range(len(y_hat))]" if p > 1 else
               "innov = y - y_hat[0]")
    b.blank()
    b.comment("Zustandsgleichung x(k+1) = Ax(k) + Bu(k)")
    b.line("u_vec = _matvec(K, [-xi for xi in x])  # u = -Kx")
    b.line("u_vec = [u_vec[i] + r for i in range(len(u_vec))]  # Vorsteuerung")
    b.line("x_new = _matvec_add(A, x, B, u_vec)")
    if L is not None:
        b.line("L_innov = _matvec_scalar(L, innov)")
        b.line("x_new = [x_new[i] + L_innov[i] for i in range(len(x_new))]")
    b.line("self.x = x_new")
    b.line("return u_vec[0] if len(u_vec) == 1 else u_vec")
    b.dedent()
    b.dedent()

    b.blank()
    b.blank()
    _write_matrix_helpers(b)

    return b.build()


# ---------------------------------------------------------------------------
# Vollständiger Regelkreis (Hauptschleife)
# ---------------------------------------------------------------------------

def gen_control_loop(
    controller_code: str,
    controller_class: str = "PIDController",
    dt_ms: int = 10,
    sensor_func: str = "read_sensor",
    actuator_func: str = "set_actuator",
    setpoint_func: str = "get_setpoint",
    cfg: Optional[CodegenConfig] = None,
) -> str:
    """Generiert eine vollständige MicroPython-Hauptschleife.

    Parameters
    ----------
    controller_code   : Bereits generierter Regler-Code (wird prepended).
    controller_class  : Klassenname des Reglers.
    dt_ms             : Abtastzeit [ms].
    sensor_func       : Name der Sensor-Lesefunktion.
    actuator_func     : Name der Aktuator-Schreibfunktion.
    setpoint_func     : Name der Sollwert-Lesefunktion.
    cfg               : CodegenConfig.
    """
    cfg = cfg or CodegenConfig()
    b = _CodeBuilder(cfg.indent)
    b.raw(controller_code)
    b.blank()
    b.blank()
    b.comment("=" * 60)
    b.comment("Hauptprogramm / Control Loop")
    b.comment("=" * 60)
    b.line("import time")
    if cfg.include_uart:
        b.line("from machine import UART")
        b.line("uart = UART(1, baudrate=115200)")
    if cfg.include_watchdog:
        b.line("from machine import WDT")
        b.line("wdt = WDT(timeout=5000)  # 5 Sekunden Watchdog")
    b.blank()
    b.blank()
    b.comment("--- Sensor / Aktuator-Stub-Funktionen (bitte anpassen!) ---")
    b.line(f"def {sensor_func}():")
    b.indent()
    b.comment("TODO: ADC oder I2C auslesen")
    b.line("return 0.0  # Platzhalterwert")
    b.dedent()
    b.blank()
    b.line(f"def {actuator_func}(value):")
    b.indent()
    b.comment("TODO: PWM, DAC oder digitalen Ausgang setzen")
    b.line("pass")
    b.dedent()
    b.blank()
    b.line(f"def {setpoint_func}():")
    b.indent()
    b.comment("TODO: Sollwert aus Potentiometer, UART, MQTT… lesen")
    b.line("return 1.0  # Platzhalterwert")
    b.dedent()
    b.blank()
    b.blank()
    b.comment("--- Initialisierung ---")
    b.line(f"controller = {controller_class}()")
    b.line(f"dt_ms = {dt_ms}  # Abtastzeit [ms]")
    b.blank()
    b.comment("--- Regelschleife ---")
    b.line("print('Regelschleife gestartet...')")
    b.line("while True:")
    b.indent()
    b.line("t_start = time.ticks_ms()")
    b.blank()
    b.comment("1. Sollwert lesen")
    b.line(f"setpoint = {setpoint_func}()")
    b.blank()
    b.comment("2. Messgröße einlesen")
    b.line(f"measurement = {sensor_func}()")
    b.blank()
    b.comment("3. Regler berechnen")
    b.line("u = controller.update(setpoint, measurement)")
    b.blank()
    b.comment("4. Stellgröße ausgeben")
    b.line(f"{actuator_func}(u)")
    b.blank()
    if cfg.include_uart:
        b.comment("5. Messdaten via UART senden (CSV)")
        b.line("uart.write(f'{setpoint:.4f},{measurement:.4f},{u:.4f}\\n')")
        b.blank()
    if cfg.include_watchdog:
        b.comment("6. Watchdog zurücksetzen")
        b.line("wdt.feed()")
        b.blank()
    b.comment("7. Abtastzeit einhalten")
    b.line("elapsed = time.ticks_diff(time.ticks_ms(), t_start)")
    b.line("sleep_ms = dt_ms - elapsed")
    b.line("if sleep_ms > 0:")
    b.indent()
    b.line("time.sleep_ms(sleep_ms)")
    b.dedent()
    b.dedent()

    return b.build()


# ---------------------------------------------------------------------------
# Vollständiges Projekt-Bundle generieren
# ---------------------------------------------------------------------------

def gen_project(
    pid_controller: PIDController,
    plant_tf: Optional[TransferFunction] = None,
    dt_ms: int = 10,
    cfg: Optional[CodegenConfig] = None,
) -> dict[str, str]:
    """Generiert ein vollständiges MicroPython-Projektbundle.

    Returns
    -------
    dict : Dateiname → Quellcode-String.
        - 'main.py'       : Hauptdatei mit Regelschleife
        - 'pid.py'        : PID-Regler-Klasse
        - 'README.md'     : Kurzdokumentation
    """
    cfg = cfg or CodegenConfig()
    dt = dt_ms / 1000.0

    # PID-Code
    pid_code = gen_pid(pid_controller, class_name="PIDController", cfg=cfg)

    # Hauptschleife
    main_code = gen_control_loop(
        controller_code="from pid import PIDController",
        controller_class="PIDController",
        dt_ms=dt_ms,
        cfg=cfg,
    )

    # README
    readme = _gen_readme(pid_controller, dt_ms, cfg)

    return {
        "main.py": main_code,
        "pid.py": pid_code,
        "README.md": readme,
    }


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _write_file_header(b: _CodeBuilder, cfg: CodegenConfig, description: str) -> None:
    """Schreibt den Datei-Header-Kommentar."""
    if cfg.add_comments:
        b.comment("=" * 65)
        b.comment(f"pylab – MicroPython-Codegen")
        b.comment(f"Beschreibung : {description}")
        b.comment(f"Zielplattform: {cfg.target}")
        b.comment(f"Generiert um : {_timestamp()}")
        b.comment("=" * 65)
        b.blank()


def _write_main_loop_pid(
    b: _CodeBuilder, class_name: str, dt: float, cfg: CodegenConfig
) -> None:
    """Schreibt einen einfachen Test-Stub für die REPL."""
    if cfg.add_comments:
        b.comment("--- Schnelltest (REPL / Simulation) ---")
    b.line("if __name__ == '__main__':")
    b.indent()
    b.line(f"ctrl = {class_name}()")
    b.line("setpoint = 1.0")
    b.line("y = 0.0")
    b.line("for i in range(200):")
    b.indent()
    b.line("u = ctrl.update(setpoint, y)")
    b.comment("Einfaches PT1-Streckenmodell für lokalen Test")
    b.line(f"y += (u - y) * {dt:.6g} / 1.0")
    b.line("print(f'{i*" + f"'{dt:.6g}'" + ":5.3f}  w={setpoint:.3f}  y={y:.4f}  u={u:.4f}')")
    b.dedent()
    b.dedent()


def _write_matrix_helpers(b: _CodeBuilder) -> None:
    """Schreibt einfache Matrix-Multiplikationsfunktionen ohne numpy."""
    b.comment("--- Matrix-Hilfsfunktionen (kein numpy erforderlich) ---")
    b.blank()
    b.line("def _matvec(A, x):")
    b.indent()
    b.comment("Matrix-Vektor-Produkt y = A * x")
    b.line("return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]")
    b.dedent()
    b.blank()
    b.line("def _matvec_add(A, x, B, u):")
    b.indent()
    b.comment("y = A*x + B*u")
    b.line("ax = _matvec(A, x)")
    b.line("bu = _matvec(B, u)")
    b.line("return [ax[i] + bu[i] for i in range(len(ax))]")
    b.dedent()
    b.blank()
    b.line("def _matvec_scalar(M, s):")
    b.indent()
    b.comment("Skalare Multiplikation: y = M * s (wenn s Skalar ist)")
    b.line("if isinstance(s, (int, float)):")
    b.indent()
    b.line("return [sum(M[i][j] * s for j in range(len(M[0]))) for i in range(len(M))]")
    b.dedent()
    b.line("return _matvec(M, s)")
    b.dedent()


def _mat_to_list(M: np.ndarray) -> str:
    """Konvertiert numpy-Matrix in Python-Listen-Literal."""
    if M.ndim == 1:
        return "[" + ", ".join(f"{v:.8g}" for v in M.tolist()) + "]"
    rows = [
        "[" + ", ".join(f"{v:.8g}" for v in row) + "]"
        for row in M.tolist()
    ]
    return "[" + ", ".join(rows) + "]"


def _fmt(v: float) -> str:
    """Formatiert float-Werte für Codegen (Inf → großer Wert)."""
    if v == float("inf"):
        return "1e18"
    if v == float("-inf"):
        return "-1e18"
    return f"{v:.8g}"


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _gen_readme(
    pid: PIDController,
    dt_ms: int,
    cfg: CodegenConfig,
) -> str:
    """Generiert eine kurze Projektdokumentation als Markdown."""
    return textwrap.dedent(f"""
    # MicroPython PID-Regelkreis
    
    Generiert von **pylab** – Zielplattform: {cfg.target}
    
    ## Dateien
    
    | Datei     | Beschreibung                          |
    |-----------|---------------------------------------|
    | `main.py` | Hauptschleife mit Regellogik           |
    | `pid.py`  | PID-Regler-Klasse                     |
    
    ## Parameter
    
    | Parameter | Wert             |
    |-----------|-----------------|
    | Kp        | {pid.Kp:.6g}    |
    | Ti        | {pid.Ti:.6g} s  |
    | Td        | {pid.Td:.6g} s  |
    | dt        | {dt_ms} ms      |
    
    ## Schnellstart
    
    1. Dateien auf das MicroPython-Device kopieren (ampy, rshell, Thonny).
    2. `read_sensor()`, `set_actuator()` und `get_setpoint()` in `main.py` anpassen.
    3. `main.py` ausführen oder als Boot-Skript eintragen.
    
    ## Hinweise
    
    - Keine externen Bibliotheken erforderlich.
    - Kompatibel mit MicroPython ≥ 1.20 (ESP32, RP2040, STM32F4).
    - Anti-Windup und D-Tiefpass sind aktiv.
    """).strip()

"""
pylabb.gui.widgets.system_editor
================================
Widget zur Eingabe und Bearbeitung von Übertragungsfunktionen,
PID-Parametern und Streckenmodellen.

Enthält:
  ``TransferFunctionEditor`` – Numerator/Denominator-Eingabe
  ``PIDEditor``              – PID-Parameterfelder mit Einstellregeln
  ``SystemEditorPanel``      – Kombiniertes Dock-Panel
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget,
    QMessageBox, QCheckBox, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.pid import (
    PIDController,
    DiscretePIDController,
    ziegler_nichols_step,
    ziegler_nichols_oscillation,
    cohen_coon,
    lambda_tuning,
)


def _parse_coeffs(text: str) -> list[float]:
    """Wandelt '1, 2, 3' oder '1 2 3' in eine Liste von Floats um."""
    text = text.replace(",", " ")
    tokens = text.split()
    return [float(t) for t in tokens]


# ---------------------------------------------------------------------------
# Übertragungsfunktions-Editor
# ---------------------------------------------------------------------------

class TransferFunctionEditor(QGroupBox):
    """Fomularfeld für Zähler und Nenner der Übertragungsfunktion.

    Signals
    -------
    tf_changed(TransferFunction) : Bei jeder validen Änderung.
    """

    tf_changed = pyqtSignal(object)  # TransferFunction

    def __init__(self, title: str = "Übertragungsfunktion", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)

        self._name_edit = QLineEdit("G")
        layout.addRow("Name:", self._name_edit)

        self._num_edit = QLineEdit("1")
        self._num_edit.setPlaceholderText("Koeffizienten, z.B.: 1 2  oder  1, 2")
        layout.addRow("Zähler:", self._num_edit)

        self._den_edit = QLineEdit("1 1")
        self._den_edit.setPlaceholderText("Koeffizienten, z.B.: 1 1  oder  1, 2, 1")
        layout.addRow("Nenner:", self._den_edit)

        self._dt_check = QCheckBox("Zeitdiskret")
        self._dt_spin = QDoubleSpinBox()
        self._dt_spin.setRange(1e-6, 100.0)
        self._dt_spin.setSingleStep(0.001)
        self._dt_spin.setValue(0.01)
        self._dt_spin.setSuffix(" s")
        self._dt_spin.setEnabled(False)
        self._dt_check.toggled.connect(self._dt_spin.setEnabled)
        dt_row = QHBoxLayout()
        dt_row.addWidget(self._dt_check)
        dt_row.addWidget(self._dt_spin)
        layout.addRow("Abtastzeit:", dt_row)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: green; font-size: 11px;")
        layout.addRow(self._status_label)

        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton("Übernehmen")
        self._apply_btn.clicked.connect(self._on_apply)
        self._reset_btn = QPushButton("Zurücksetzen")
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._reset_btn)
        layout.addRow(btn_row)

    def _on_apply(self) -> None:
        try:
            num = _parse_coeffs(self._num_edit.text())
            den = _parse_coeffs(self._den_edit.text())
            dt = self._dt_spin.value() if self._dt_check.isChecked() else None
            name = self._name_edit.text() or "G"
            tf = TransferFunction(num, den, dt=dt, name=name)
            poles = tf.poles()
            self._status_label.setText(
                f"OK – Ordnung {len(den)-1}, Pole: {len(poles)}, stabil: {tf.is_stable()}"
            )
            self._status_label.setStyleSheet("color: green; font-size: 11px;")
            self.tf_changed.emit(tf)
        except Exception as ex:
            self._status_label.setText(f"Fehler: {ex}")
            self._status_label.setStyleSheet("color: red; font-size: 11px;")

    def _on_reset(self) -> None:
        self._num_edit.setText("1")
        self._den_edit.setText("1 1")
        self._name_edit.setText("G")
        self._dt_check.setChecked(False)

    def get_tf(self) -> Optional[TransferFunction]:
        """Liest die aktuellen Felder aus und konstruiert eine TransferFunction.

        Returns
        -------
        TransferFunction | None
            Das aus den Eingabefeldern aufgebaute Übertragungsfunktionsobjekt,
            oder ``None`` wenn die Koeffizienten nicht parsebar sind.
        """
        try:
            num = _parse_coeffs(self._num_edit.text())
            den = _parse_coeffs(self._den_edit.text())
            dt = self._dt_spin.value() if self._dt_check.isChecked() else None
            return TransferFunction(num, den, dt=dt, name=self._name_edit.text() or "G")
        except Exception:
            return None

    def set_tf(self, tf: TransferFunction) -> None:
        """Befüllt die Eingabefelder mit den Werten einer TransferFunction.

        Parameters
        ----------
        tf : TransferFunction
            Quellobjekt. Zähler- und Nenner-Koeffizienten werden als
            Leerzeichen-getrennte Zeichenketten eingetragen.
            Falls ``tf.dt`` gesetzt ist, wird der Zeitdiskret-Modus aktiviert.
        """
        num_str = " ".join(f"{v:.6g}" for v in tf.num)
        den_str = " ".join(f"{v:.6g}" for v in tf.den)
        self._num_edit.setText(num_str)
        self._den_edit.setText(den_str)
        if tf.name:
            self._name_edit.setText(tf.name)
        if tf.dt is not None:
            self._dt_check.setChecked(True)
            self._dt_spin.setValue(tf.dt)


# ---------------------------------------------------------------------------
# PID-Editor
# ---------------------------------------------------------------------------

class PIDEditor(QGroupBox):
    """Formularfelder für PID-Parameter plus Einstellregel-Auswahl.

    Signals
    -------
    pid_changed(PIDController) : Beim Anwenden der Parameter.
    """

    pid_changed = pyqtSignal(object)  # PIDController

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("PID-Regler", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Parameter-Felder
        param_group = QGroupBox("Parameter")
        form = QFormLayout(param_group)

        def _spin(lo=0.0, hi=1e6, step=0.1, val=1.0, suffix=""):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setValue(val)
            s.setDecimals(4)
            if suffix:
                s.setSuffix(f" {suffix}")
            return s

        self._kp_spin = _spin(0, 1e6, 0.1, 1.0)
        self._ti_spin = _spin(0, 1e6, 0.1, 0.0, "s")
        self._td_spin = _spin(0, 1e6, 0.01, 0.0, "s")
        self._n_spin  = _spin(1, 1000, 1, 20)
        self._dt_spin = _spin(1e-6, 10, 0.001, 0.01, "s")

        form.addRow("Kp:", self._kp_spin)
        form.addRow("Ti [s]:", self._ti_spin)
        form.addRow("Td [s]:", self._td_spin)
        form.addRow("N (D-Filter):", self._n_spin)
        form.addRow("dt (Abtastzeit):", self._dt_spin)
        layout.addWidget(param_group)

        # Einstellregeln
        tuning_group = QGroupBox("Einstellregeln (Ziegler-Nichols / Cohen-Coon)")
        tuning_form = QFormLayout(tuning_group)
        self._rule_combo = QComboBox()
        self._rule_combo.addItems(["ZN-Sprung", "ZN-Schwingung", "Cohen-Coon", "Lambda"])
        tuning_form.addRow("Methode:", self._rule_combo)

        self._K_spin  = _spin(0.001, 1e6, 0.1, 1.0)
        self._T_spin  = _spin(0.001, 1e6, 0.1, 1.0, "s")
        self._L_spin  = _spin(0.0, 1e6, 0.01, 0.1, "s")
        self._type_combo = QComboBox()
        self._type_combo.addItems(["PID", "PI", "P"])
        tuning_form.addRow("K (Strecke):", self._K_spin)
        tuning_form.addRow("T (Zeitkonstante):", self._T_spin)
        tuning_form.addRow("L (Totzeit):", self._L_spin)
        tuning_form.addRow("Reglertyp:", self._type_combo)

        tune_btn = QPushButton("Parameter berechnen")
        tune_btn.clicked.connect(self._on_tune)
        tuning_form.addRow(tune_btn)
        layout.addWidget(tuning_group)

        # Anwendungs-Buttons
        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Übernehmen")
        apply_btn.clicked.connect(self._on_apply)
        reset_btn = QPushButton("Zurücksetzen")
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(reset_btn)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setStyleSheet("color: green; font-size: 11px;")
        layout.addWidget(self._status)

    def _on_tune(self) -> None:
        """Berechnet PID-Parameter anhand der gewählten Einstellregel."""
        method = self._rule_combo.currentText()
        K = self._K_spin.value()
        T = self._T_spin.value()
        L = self._L_spin.value()
        ct = self._type_combo.currentText()

        try:
            if method == "ZN-Sprung":
                pid = ziegler_nichols_step(K, T, L, ct)
            elif method == "ZN-Schwingung":
                # Ku ≈ K, Tu ≈ T (vereinfacht für Demonstration)
                pid = ziegler_nichols_oscillation(Ku=K, Tu=T, controller_type=ct)
            elif method == "Cohen-Coon":
                pid = cohen_coon(K, T, L, ct)
            else:
                pid = lambda_tuning(K, T, L)
            self._set_from_pid(pid)
            self._status.setText(f"Berechnet: {method}")
            self._status.setStyleSheet("color: green; font-size: 11px;")
        except Exception as ex:
            self._status.setText(f"Fehler: {ex}")
            self._status.setStyleSheet("color: red; font-size: 11px;")

    def _set_from_pid(self, pid: PIDController) -> None:
        self._kp_spin.setValue(pid.Kp)
        self._ti_spin.setValue(pid.Ti if pid.Ti > 0 else 0.0)
        self._td_spin.setValue(pid.Td)
        self._n_spin.setValue(pid.N)

    def _on_apply(self) -> None:
        pid = self.get_pid()
        self.pid_changed.emit(pid)
        self._status.setText(
            f"PID übernommen: Kp={pid.Kp:.4g}, Ti={pid.Ti:.4g}, Td={pid.Td:.4g}"
        )

    def _on_reset(self) -> None:
        self._kp_spin.setValue(1.0)
        self._ti_spin.setValue(0.0)
        self._td_spin.setValue(0.0)
        self._n_spin.setValue(20.0)

    def get_pid(self) -> PIDController:
        """Liest die Spin-Boxen aus und erstellt einen PIDController.

        Returns
        -------
        PIDController
            Aktueller Regler mit Kp, Ti, Td, N aus den Eingabefeldern.
        """
        return PIDController(
            Kp=self._kp_spin.value(),
            Ti=self._ti_spin.value(),
            Td=self._td_spin.value(),
            N=self._n_spin.value(),
        )

    def get_discrete_pid(self) -> DiscretePIDController:
        """Erstellt den zeitdiskreten PID aus den aktuellen Eingabewerten.

        Returns
        -------
        DiscretePIDController
            Diskretisierter Regler mit Kp, Ti, Td, N und Abtastzeit dt
            aus den Eingabefeldern.
        """
        return DiscretePIDController(
            Kp=self._kp_spin.value(),
            Ti=self._ti_spin.value() or float("inf"),
            Td=self._td_spin.value(),
            dt=self._dt_spin.value(),
            N=self._n_spin.value(),
        )


# ---------------------------------------------------------------------------
# System-Editor-Panel (Tab-Widget mit ÜTF + PID)
# ---------------------------------------------------------------------------

class SystemEditorPanel(QWidget):
    """Haupt-Panel mit Tabs für Strecke, Regler und Standardsysteme.

    Kapselt einen :class:`TransferFunctionEditor` (Tab *Strecke*) und
    einen :class:`PIDEditor` (Tab *PID-Regler*) sowie Schnellauswahl-
    Schaltflächen für gängige Standardstreckenmodelle.

    Parameters
    ----------
    parent : QWidget, optional
        Übergeordnetes Widget.

    Attributes
    ----------
    tf_editor : TransferFunctionEditor
        Direkt zugreifbarer ÜTF-Editor (Tab 1).
    pid_editor : PIDEditor
        Direkt zugreifbarer PID-Editor (Tab 2).

    Signals
    -------
    plant_changed(TransferFunction) : Wird durchgereicht von :attr:`tf_editor`.
    controller_changed(PIDController) : Wird durchgereicht von :attr:`pid_editor`.
    """

    plant_changed = pyqtSignal(object)
    controller_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        tabs = QTabWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(tabs)

        # Tab 1: Strecke
        self._tf_editor = TransferFunctionEditor("Strecke G(s)")
        self._tf_editor.tf_changed.connect(self.plant_changed)
        tabs.addTab(self._tf_editor, "Strecke")

        # Tab 2: Regler
        self._pid_editor = PIDEditor()
        self._pid_editor.pid_changed.connect(self.controller_changed)
        tabs.addTab(self._pid_editor, "PID-Regler")

        # Tab 3: Standardsysteme
        std = self._build_standard_systems()
        tabs.addTab(std, "Standardsysteme")

    def _build_standard_systems(self) -> QWidget:
        """Panel mit Schaltflächen für häufig verwendete Streckenmodelle."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Schnellauswahl Standardstrecken:"))

        examples = [
            ("PT1  K=1, T=1",              [1],       [1, 1]),
            ("PT2  ωn=1, ζ=0.7",           [1],       [1, 1.4, 1]),
            ("Integrierend K=1",            [1],       [1, 0]),
            ("2. Ord. instabil",            [1],       [1, -1, -2]),
            ("Minimalphase K=1, T=1, L=0.1",[1],      [1, 1]),
        ]

        for label, num, den in examples:
            btn = QPushButton(label)
            _num, _den = num[:], den[:]

            def _cb(checked=False, n=_num, d=_den):
                tf = TransferFunction(n, d, name="G")
                self._tf_editor.set_tf(tf)
                self.plant_changed.emit(tf)

            btn.clicked.connect(_cb)
            layout.addWidget(btn)

        layout.addStretch()
        return w

    @property
    def tf_editor(self) -> TransferFunctionEditor:
        """Der Übertragungsfunktions-Editor (Tab *Strecke*).

        Returns
        -------
        TransferFunctionEditor
            Internes :class:`TransferFunctionEditor`-Objekt.
        """
        return self._tf_editor

    @property
    def pid_editor(self) -> PIDEditor:
        """Der PID-Regler-Editor (Tab *PID-Regler*).

        Returns
        -------
        PIDEditor
            Internes :class:`PIDEditor`-Objekt.
        """
        return self._pid_editor

    def get_plant(self) -> Optional[TransferFunction]:
        """Delegiert an :meth:`TransferFunctionEditor.get_tf`.

        Returns
        -------
        TransferFunction | None
            Aktuelle Strecken-ÜTF oder ``None`` bei Ungültigkeit.
        """
        return self._tf_editor.get_tf()

    def get_pid(self) -> PIDController:
        """Delegiert an :meth:`PIDEditor.get_pid`.

        Returns
        -------
        PIDController
            Aktueller PID-Regler aus dem Editor.
        """
        return self._pid_editor.get_pid()

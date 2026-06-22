"""
pylabb.gui.widgets.codegen_widget
==================================
Widget für die MicroPython-Codegenerierung.

Zeigt generierten Code in einem Syntax-Highlight-fähigen Editor an und
bietet Optionen für:
  - PID-Code generieren
  - Digitaler Filter (diskrete ÜTF)
  - Vollständige Hauptschleife
  - Projektbundle exportieren
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPlainTextEdit, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QDoubleSpinBox, QFormLayout,
    QCheckBox, QSplitter, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat, QColor

from pylabb.codegen.micropython import (
    CodegenConfig,
    gen_pid,
    gen_digital_filter,
    gen_control_loop,
    gen_project,
)
from pylabb.control.pid import PIDController


# ---------------------------------------------------------------------------
# Einfacher Python-Syntax-Highlighter für den Code-Editor
# ---------------------------------------------------------------------------

class _PythonHighlighter(QSyntaxHighlighter):
    """Minimaler Python-Syntax-Highlighter."""

    KEYWORDS = {
        "def", "class", "if", "else", "elif", "for", "while", "return",
        "import", "from", "True", "False", "None", "in", "not", "and",
        "or", "pass", "self", "print",
    }

    def __init__(self, doc) -> None:
        super().__init__(doc)
        self._kw_fmt = QTextCharFormat()
        self._kw_fmt.setForeground(QColor("#0000cc"))
        self._kw_fmt.setFontWeight(700)

        self._comment_fmt = QTextCharFormat()
        self._comment_fmt.setForeground(QColor("#5c7a5c"))
        self._comment_fmt.setFontItalic(True)

        self._string_fmt = QTextCharFormat()
        self._string_fmt.setForeground(QColor("#a31515"))

        self._number_fmt = QTextCharFormat()
        self._number_fmt.setForeground(QColor("#097e09"))

    def highlightBlock(self, text: str) -> None:
        import re
        # Kommentare
        m = re.search(r"#.*$", text)
        if m:
            self.setFormat(m.start(), len(m.group()), self._comment_fmt)
            text = text[:m.start()]  # Rest ignorieren

        # Strings
        for m in re.finditer(r'"[^"]*"|\'[^\']*\'', text):
            self.setFormat(m.start(), m.end() - m.start(), self._string_fmt)

        # Zahlen
        for m in re.finditer(r"\b\d+\.?\d*([eE][+-]?\d+)?\b", text):
            self.setFormat(m.start(), m.end() - m.start(), self._number_fmt)

        # Schlüsselwörter
        for m in re.finditer(r"\b\w+\b", text):
            if m.group() in self.KEYWORDS:
                self.setFormat(m.start(), m.end() - m.start(), self._kw_fmt)


# ---------------------------------------------------------------------------
# Codegen-Widget
# ---------------------------------------------------------------------------

class CodegenWidget(QWidget):
    """Panel zur MicroPython-Codegenerierung für Embedded-Systeme.

    Ermöglicht die Generierung von MicroPython-Quellcode für drei Modi:

    * **PID-Regler** – diskretisierter Positionsalgorithmus.
    * **Digitaler Filter** – aus einer kontinuierlichen ÜTF diskretisiert.
    * **Vollständige Hauptschleife** – inklusive Interrupt-Timer und UART.

    Parameters
    ----------
    parent : QWidget, optional
        Übergeordnetes Widget.

    Attributes
    ----------
    code_generated : pyqtSignal(str)
        Wird nach jeder erfolgreichen Codegenerierung emittiert.

    Signals
    -------
    code_generated(str) : Sobald neuer Code generiert wurde.
    """

    code_generated = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pid: PIDController = PIDController()
        self._plant_tf = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI-Aufbau
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Obere Hälfte: Optionen
        options_panel = self._build_options()
        splitter.addWidget(options_panel)

        # Untere Hälfte: Code-Editor
        self._code_edit = QPlainTextEdit()
        self._code_edit.setFont(QFont("Courier New", 10))
        self._code_edit.setReadOnly(False)
        self._code_edit.setPlaceholderText("Generierter MicroPython-Code erscheint hier …")
        _PythonHighlighter(self._code_edit.document())
        splitter.addWidget(self._code_edit)

        splitter.setSizes([200, 400])
        layout.addWidget(splitter)

        # Button-Leiste unten
        btn_row = QHBoxLayout()
        self._gen_btn = QPushButton("Code generieren")
        self._gen_btn.clicked.connect(self._on_generate)
        self._copy_btn = QPushButton("In Zwischenablage")
        self._copy_btn.clicked.connect(self._on_copy)
        self._export_btn = QPushButton("Projektbundle exportieren …")
        self._export_btn.clicked.connect(self._on_export_project)
        btn_row.addWidget(self._gen_btn)
        btn_row.addWidget(self._copy_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

    def _build_options(self) -> QWidget:
        panel = QGroupBox("Codegen-Optionen")
        form = QFormLayout(panel)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "PID-Regler",
            "Digitaler Filter (diskrete ÜTF)",
            "Vollständige Hauptschleife",
        ])
        form.addRow("Modus:", self._mode_combo)

        self._dt_spin = QDoubleSpinBox()
        self._dt_spin.setRange(0.001, 10.0)
        self._dt_spin.setSingleStep(0.001)
        self._dt_spin.setValue(0.01)
        self._dt_spin.setSuffix(" s")
        form.addRow("Abtastzeit dt:", self._dt_spin)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["tustin", "zoh", "euler"])
        form.addRow("Diskretisierungsmethode:", self._method_combo)

        self._target_combo = QComboBox()
        self._target_combo.addItems([
            "ESP32 / RP2040 / STM32",
            "ESP8266",
            "Pyboard",
            "Generic MicroPython",
        ])
        form.addRow("Zielplattform:", self._target_combo)

        self._comments_check = QCheckBox("Kommentare einschließen")
        self._comments_check.setChecked(True)
        form.addRow(self._comments_check)

        self._uart_check = QCheckBox("UART-Debug-Ausgabe")
        form.addRow(self._uart_check)

        self._watchdog_check = QCheckBox("Watchdog-Timer")
        form.addRow(self._watchdog_check)

        return panel

    # ------------------------------------------------------------------
    # Aktionen
    # ------------------------------------------------------------------

    def _on_generate(self) -> None:
        cfg = self._make_config()
        mode = self._mode_combo.currentText()
        dt = self._dt_spin.value()

        try:
            if mode == "PID-Regler":
                code = gen_pid(self._pid, cfg=cfg)
            elif mode == "Digitaler Filter (diskrete ÜTF)":
                if self._plant_tf is None:
                    QMessageBox.warning(self, "Kein System", "Bitte zuerst eine Strecke definieren.")
                    return
                code = gen_digital_filter(
                    self._plant_tf,
                    dt=dt,
                    method=self._method_combo.currentText(),
                    cfg=cfg,
                )
            else:
                pid_code = gen_pid(self._pid, cfg=cfg)
                code = gen_control_loop(
                    controller_code=pid_code,
                    dt_ms=int(dt * 1000),
                    cfg=cfg,
                )
            self._code_edit.setPlainText(code)
            self.code_generated.emit(code)
        except Exception as ex:
            QMessageBox.critical(self, "Fehler bei Codegenerierung", str(ex))

    def _on_copy(self) -> None:
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._code_edit.toPlainText())

    def _on_export_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Projektverzeichnis wählen"
        )
        if not directory:
            return
        cfg = self._make_config()
        dt_ms = int(self._dt_spin.value() * 1000)
        try:
            files = gen_project(self._pid, plant_tf=self._plant_tf, dt_ms=dt_ms, cfg=cfg)
            for filename, content in files.items():
                path = os.path.join(directory, filename)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            QMessageBox.information(
                self, "Export erfolgreich",
                f"Projektbundle nach {directory} exportiert:\n"
                + "\n".join(f"  • {fn}" for fn in files),
            )
        except Exception as ex:
            QMessageBox.critical(self, "Export fehlgeschlagen", str(ex))

    def _make_config(self) -> CodegenConfig:
        return CodegenConfig(
            add_comments=self._comments_check.isChecked(),
            target=self._target_combo.currentText(),
            include_uart=self._uart_check.isChecked(),
            include_watchdog=self._watchdog_check.isChecked(),
        )

    # ------------------------------------------------------------------
    # Öffentliche Setter
    # ------------------------------------------------------------------

    def set_pid(self, pid: PIDController) -> None:
        """Aktualisiert den intern gespeicherten PID-Regler.

        Wird vom :class:`~pylabb.gui.main_window.MainWindow` aufgerufen,
        wenn der Nutzer den Regler im :class:`~.system_editor.SystemEditorPanel`
        ändert.

        Parameters
        ----------
        pid : PIDController
            Neuer PID-Regler, der bei der nächsten Codegenerierung verwendet wird.
        """
        self._pid = pid

    def set_plant_tf(self, tf) -> None:
        """Setzt die Strecken-ÜTF für den Digitalfilter-Modus.

        Parameters
        ----------
        tf : TransferFunction
            Strecke G(s). Wird für den Modus *Digitaler Filter (diskrete ÜTF)*
            als Vorlage für die Diskretisierung verwendet.
        """
        self._plant_tf = tf

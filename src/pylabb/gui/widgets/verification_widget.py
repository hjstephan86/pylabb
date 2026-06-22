"""
pylabb.gui.widgets.verification_widget
=======================================
Widget zur Äquivalenzüberprüfung zweier Regelkreise per Subgraph Algorithmus.

Aufbau
------
┌──────────────────────────────────────────────────────┐
│  Regelkreis A                │  Regelkreis B          │
│  ┌────────────────────────┐  │  ┌────────────────────┐│
│  │ Strecke G(s)  [Zähler] │  │  │ Strecke G(s) ...   ││
│  │               [Nenner] │  │  │                    ││
│  │ Regler C(s)   [Zähler] │  │  │ Regler C(s) ...    ││
│  │               [Nenner] │  │  │                    ││
│  └────────────────────────┘  │  └────────────────────┘│
├──────────────────────────────────────────────────────┤
│              [ Äquivalenz prüfen ]                    │
├──────────────────────────────────────────────────────┤
│  Ergebnis-Panel                                       │
│  • Entscheidungs-Badge                                │
│  • Adjazenzmatrizen A und B (tabellarisch)            │
│  • Textuelle Zusammenfassung                          │
└──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QLineEdit,
)

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.verification import VerificationResult, verify_loops


# ---------------------------------------------------------------------------
# Hilfsklassen
# ---------------------------------------------------------------------------

def _parse_coeffs(text: str) -> list[float]:
    """Wandelt '1, 2, 3' oder '1 2 3' in eine Liste von Floats um."""
    text = text.replace(",", " ")
    tokens = text.split()
    return [float(t) for t in tokens]


class _DecisionBadge(QLabel):
    """Farbiges Label das die Entscheidung des Algorithmus anzeigt."""

    _COLORS = {
        "equal": ("#27ae60", "ÄQUIVALENT"),
        "equal_keep_a": ("#27ae60", "ÄQUIVALENT (A bevorzugt)"),
        "equal_keep_b": ("#27ae60", "ÄQUIVALENT (B bevorzugt)"),
        "keep_a": ("#2980b9", "A ENTHÄLT B"),
        "keep_b": ("#8e44ad", "B ENTHÄLT A"),
        "keep_both": ("#e67e22", "STRUKTURELL VERSCHIEDEN"),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("  –  ", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.setFont(font)
        self._set_neutral()

    def _set_neutral(self) -> None:
        self.setText("  –  ")
        self.setStyleSheet(
            "background-color:#95a5a6;color:white;border-radius:6px;padding:6px 12px;"
        )

    def update_decision(self, decision: str) -> None:
        key = decision.lower()
        color, text = self._COLORS.get(key, ("#95a5a6", decision.upper()))
        self.setText(f"  {text}  ")
        self.setStyleSheet(
            f"background-color:{color};color:white;border-radius:6px;padding:6px 12px;"
        )


class _MatrixView(QGroupBox):
    """Zeigt eine quadratische Adjazenzmatrix als QTableWidget an."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMaximumHeight(200)
        layout.addWidget(self._table)

    def set_matrix(self, matrix: np.ndarray) -> None:
        n = matrix.shape[0]
        self._table.setRowCount(n)
        self._table.setColumnCount(n)
        self._table.setHorizontalHeaderLabels([f"x{i}" for i in range(n)])
        self._table.setVerticalHeaderLabels([f"x{i}" for i in range(n)])

        for i in range(n):
            for j in range(n):
                val = int(matrix[i, j])
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if val == 1:
                    item.setBackground(QColor("#d5e8d4"))
                self._table.setItem(i, j, item)

        self._table.resizeColumnsToContents()
        self._table.resizeRowsToContents()

    def clear_matrix(self) -> None:
        self._table.setRowCount(0)
        self._table.setColumnCount(0)


class _LoopInputPanel(QGroupBox):
    """Eingabeformular für Strecke G(s) und Regler C(s) eines Regelkreises."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)

        # Strecke G(s)
        self._g_num = QLineEdit("1")
        self._g_num.setPlaceholderText("z.B. 1")
        self._g_den = QLineEdit("1 2 1")
        self._g_den.setPlaceholderText("z.B. 1 2 1")
        layout.addRow("<b>G(s)</b> – Zähler:", self._g_num)
        layout.addRow("G(s) – Nenner:", self._g_den)

        # Regler C(s)
        self._c_num = QLineEdit("1 1")
        self._c_num.setPlaceholderText("z.B. 1 1")
        self._c_den = QLineEdit("1")
        self._c_den.setPlaceholderText("z.B. 1")
        layout.addRow("<b>C(s)</b> – Zähler:", self._c_num)
        layout.addRow("C(s) – Nenner:", self._c_den)

    def get_plant(self) -> TransferFunction:
        """Liest G(s) aus den Eingabefeldern."""
        num = _parse_coeffs(self._g_num.text())
        den = _parse_coeffs(self._g_den.text())
        return TransferFunction(num, den, name="G")

    def get_controller(self) -> TransferFunction:
        """Liest C(s) aus den Eingabefeldern."""
        num = _parse_coeffs(self._c_num.text())
        den = _parse_coeffs(self._c_den.text())
        return TransferFunction(num, den, name="C")

    def set_plant(self, tf: TransferFunction) -> None:
        self._g_num.setText(" ".join(str(c) for c in tf.num.real))
        self._g_den.setText(" ".join(str(c) for c in tf.den.real))

    def set_controller(self, tf: TransferFunction) -> None:
        self._c_num.setText(" ".join(str(c) for c in tf.num.real))
        self._c_den.setText(" ".join(str(c) for c in tf.den.real))


# ---------------------------------------------------------------------------
# Haupt-Widget
# ---------------------------------------------------------------------------

class VerificationWidget(QWidget):
    """Vollständiges Widget zur Äquivalenzüberprüfung zweier Regelkreise.

    Signals
    -------
    verification_done(VerificationResult) : Nach jeder Überprüfung.
    """

    verification_done = pyqtSignal(object)  # VerificationResult

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_result: Optional[VerificationResult] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI-Aufbau
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Kopfzeile
        header = QHBoxLayout()
        title = QLabel("<b>Regelkreis-Äquivalenzprüfung (Subgraph)</b>")
        title.setStyleSheet("font-size: 13px;")
        header.addWidget(title)
        header.addStretch()
        main_layout.addLayout(header)

        # Eingabe-Panels (A und B nebeneinander)
        input_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._loop_A = _LoopInputPanel("Regelkreis A")
        self._loop_B = _LoopInputPanel("Regelkreis B")
        input_splitter.addWidget(self._loop_A)
        input_splitter.addWidget(self._loop_B)
        input_splitter.setSizes([400, 400])
        main_layout.addWidget(input_splitter)

        # Buttons
        btn_row = QHBoxLayout()
        self._verify_btn = QPushButton("Äquivalenz prüfen")
        self._verify_btn.setStyleSheet(
            "QPushButton { background-color: #2c3e50; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #34495e; }"
        )
        self._verify_btn.clicked.connect(self._on_verify)
        self._clear_btn = QPushButton("Ergebnis löschen")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_row.addStretch()
        btn_row.addWidget(self._verify_btn)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch()
        main_layout.addLayout(btn_row)

        # Ergebnis-Badge
        badge_row = QHBoxLayout()
        badge_row.addStretch()
        self._badge = _DecisionBadge()
        badge_row.addWidget(self._badge)
        badge_row.addStretch()
        main_layout.addLayout(badge_row)

        # Matrizen-Ansichten
        matrix_row = QHBoxLayout()
        self._mat_view_A = _MatrixView("Adjazenzmatrix  G_A")
        self._mat_view_B = _MatrixView("Adjazenzmatrix  G_B")
        matrix_row.addWidget(self._mat_view_A)
        matrix_row.addWidget(self._mat_view_B)
        main_layout.addLayout(matrix_row)

        # Textuelle Zusammenfassung
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumHeight(130)
        self._result_text.setPlaceholderText(
            "Ergebnis der Äquivalenzprüfung erscheint hier …"
        )
        main_layout.addWidget(self._result_text)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_verify(self) -> None:
        """Führt die Äquivalenzprüfung aus und zeigt das Ergebnis."""
        try:
            plant_A = self._loop_A.get_plant()
            ctrl_A  = self._loop_A.get_controller()
            plant_B = self._loop_B.get_plant()
            ctrl_B  = self._loop_B.get_controller()
        except ValueError as exc:
            QMessageBox.warning(self, "Eingabefehler", f"Ungültige Koeffizienten:\n{exc}")
            return

        try:
            result = verify_loops(
                plant_A, ctrl_A,
                plant_B, ctrl_B,
                label_A="Regelkreis A",
                label_B="Regelkreis B",
            )
        except ImportError as exc:
            QMessageBox.critical(
                self,
                "Fehlende Abhängigkeit",
                str(exc),
            )
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Verifikationsfehler", str(exc))
            return

        self._last_result = result
        self._display_result(result)
        self.verification_done.emit(result)

    def _on_clear(self) -> None:
        self._badge._set_neutral()
        self._mat_view_A.clear_matrix()
        self._mat_view_B.clear_matrix()
        self._result_text.clear()
        self._last_result = None

    # ------------------------------------------------------------------
    # Ergebnis anzeigen
    # ------------------------------------------------------------------

    def _display_result(self, result: VerificationResult) -> None:
        # Badge
        self._badge.update_decision(result.decision)

        # Matrizen
        self._mat_view_A.set_matrix(result.matrix_A)
        self._mat_view_B.set_matrix(result.matrix_B)

        # Textuelle Ausgabe
        lines: list[str] = [
            f"<b>Entscheidung:</b> {result.decision}",
            f"<b>Zusammenfassung:</b> {result.summary}",
            "",
            f"<b>Offene Kreise:</b>",
            f"&nbsp;&nbsp;L_A(s) = {result.open_loop_A.name} "
            f"[Ordnung {len(result.open_loop_A.den) - 1}]",
            f"&nbsp;&nbsp;L_B(s) = {result.open_loop_B.name} "
            f"[Ordnung {len(result.open_loop_B.den) - 1}]",
            "",
            f"<b>Adjazenzmatrix A</b> ({result.matrix_A.shape[0]}×{result.matrix_A.shape[0]}, "
            f"{int(result.matrix_A.sum())} Kanten):",
            f"<pre>{result.matrix_A}</pre>",
            f"<b>Adjazenzmatrix B</b> ({result.matrix_B.shape[0]}×{result.matrix_B.shape[0]}, "
            f"{int(result.matrix_B.sum())} Kanten):",
            f"<pre>{result.matrix_B}</pre>",
        ]
        if result.kept_matrix is not None:
            lines += [
                f"<b>Bevorzugte Matrix:</b>",
                f"<pre>{result.kept_matrix}</pre>",
            ]
        self._result_text.setHtml("<br>".join(lines))

    # ------------------------------------------------------------------
    # Externe Schnittstelle
    # ------------------------------------------------------------------

    def set_loop_A(self, plant: TransferFunction, controller: TransferFunction) -> None:
        """Setzt Strecke und Regler von Regelkreis A (aus dem Hauptfenster)."""
        self._loop_A.set_plant(plant)
        self._loop_A.set_controller(controller)

    def set_loop_B(self, plant: TransferFunction, controller: TransferFunction) -> None:
        """Setzt Strecke und Regler von Regelkreis B."""
        self._loop_B.set_plant(plant)
        self._loop_B.set_controller(controller)

    @property
    def last_result(self) -> Optional[VerificationResult]:
        """Letztes vollständiges Prüfergebnis.

        Returns
        -------
        VerificationResult | None
            Das zuletzt durch :func:`~pylabb.control.verification.verify_closed_loop`
            ermittelte Ergebnis, oder ``None`` falls noch keine Prüfung ausgeführt
            oder das Panel zurückgesetzt wurde.
        """
        return self._last_result

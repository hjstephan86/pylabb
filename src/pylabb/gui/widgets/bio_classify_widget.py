"""
pylabb.gui.widgets.bio_classify_widget
========================================
Widget zur Klassifikation biologischer Regelkreis-Äquivalenzklassen.

Aufbau
------
┌──────────────────────────────────────────────────────┐
│  Strecke G(s) + Regler C(s) Eingabe                  │
├──────────────────────────────────────────────────────┤
│  Topologie-Flags (Feedforward, Kaskade, MIMO, …)     │
├──────────────────────────────────────────────────────┤
│              [ Klassifizieren ]                      │
├──────────────────────────────────────────────────────┤
│  Ergebnis-Panel                                      │
│  • Klassen-Badge (farbig)                            │
│  • Konfidenz-Label                                   │
│  • Subgraph-Kette                                    │
│  • Adjazenzmatrix (tabellarisch)                     │
│  • Erweiterungen (Textfeld)                          │
└──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.bio_classify import (
    BioClass,
    BioClassResult,
    classify_loop,
    get_extensions,
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _parse_coeffs(text: str) -> list[float]:
    """Wandelt '1, 2, 3' oder '1 2 3' in eine Liste von Floats um."""
    text = text.replace(",", " ")
    tokens = text.split()
    return [float(t) for t in tokens]


# ---------------------------------------------------------------------------
# Hilfsklassen
# ---------------------------------------------------------------------------

class _BioBadge(QLabel):
    """Farbiges Badge für die biologische Klasse."""

    _COLORS: dict[str, tuple[str, str]] = {
        "SEESTERN": ("#3498db", "Klasse I – Seestern"),
        "PUPILLE": ("#9b59b6", "Klasse II – Pupille"),
        "KLEINHIRN": ("#1abc9c", "Klasse III – Kleinhirn"),
        "HERZ": ("#e74c3c", "Klasse IV – Herz"),
        "ALBATROS": ("#f39c12", "Klasse V – Albatros"),
        "MIMOSE": ("#27ae60", "Klasse VI – Mimose"),
        "TINTENFISCH": ("#2c3e50", "Klasse VII – Tintenfisch"),
        "UNCLASSIFIED": ("#95a5a6", "Nicht klassifiziert"),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("  –  ", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.setFont(font)
        self._set_neutral()

    def _set_neutral(self) -> None:
        self.setText("  –  ")
        self.setStyleSheet(
            "background-color:#95a5a6;color:white;border-radius:6px;padding:8px 16px;"
        )

    def update_class(self, bio_class: BioClass, confidence: float) -> None:
        key = bio_class.name.upper()
        color, label = self._COLORS.get(key, ("#95a5a6", key))
        self.setText(f"  {label}  (Konfidenz: {confidence:.0%})  ")
        self.setStyleSheet(
            f"background-color:{color};color:white;border-radius:6px;padding:8px 16px;"
        )


class _MatrixView(QGroupBox):
    """Zeigt eine quadratische Adjazenzmatrix als QTableWidget an."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMaximumHeight(180)
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


# ---------------------------------------------------------------------------
# Haupt-Widget
# ---------------------------------------------------------------------------

class BioClassifyWidget(QWidget):
    """Vollständiges Widget zur Klassifikation biologischer Regelkreis-Äquivalenzklassen.

    Das Widget ermöglicht die interaktive Klassifikation eines Regelkreises
    in eine der sieben biologischen Äquivalenzklassen (Seestern bis Tintenfisch)
    gemäß der Theorie aus Kapitel 5 der PyLab-Arbeit.

    Parameters
    ----------
    parent : QWidget, optional
        Übergeordnetes Widget.

    Attributes
    ----------
    classification_done : pyqtSignal(BioClassResult)
        Wird nach jeder erfolgreichen Klassifikation emittiert.
    last_result : BioClassResult | None
        Letztes Klassifikationsergebnis oder ``None``.

    GUI-Elemente
    ------------
    Eingabe-Panel
        Zähler und Nenner für G(s) und C(s) als Leerzeichen- oder
        Komma-getrennte Koeffizienten.
    Topologie-Flags
        Checkboxen: *Vorsteuerung*, *Kaskadenregelung*, *Full-State-Feedback*,
        *MIMO*, *Hysterese* (+ Band), *Störgrößen-Kanal*, sowie
        numerischer Konfidenz-Schwellwert.
    Klassifizieren-Button
        Löst :func:`~pylabb.control.bio_classify.classify_loop` aus.
    Ergebnis-Panel
        Farbiges Klassen-Badge mit Konfidenz, Subgraph-Kette,
        Adjazenzmatrix und HTML-Erweiterungsvorschläge.

    Signals
    -------
    classification_done(BioClassResult) : Nach jeder Klassifikation.
    """

    classification_done = pyqtSignal(object)  # BioClassResult

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_result: Optional[BioClassResult] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI-Aufbau
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Kopfzeile
        title = QLabel("<b>Biologische Äquivalenzklassen-Klassifikation</b>")
        title.setStyleSheet("font-size: 13px;")
        main_layout.addWidget(title)

        # Eingabe-Panel
        input_group = QGroupBox("Regelkreis – Strecke G(s) und Regler C(s)")
        input_form = QFormLayout(input_group)

        self._g_num = QLineEdit("1")
        self._g_num.setPlaceholderText("z.B. 1")
        self._g_den = QLineEdit("1 1.4 1")
        self._g_den.setPlaceholderText("z.B. 1 1.4 1")
        input_form.addRow("<b>G(s)</b> – Zähler:", self._g_num)
        input_form.addRow("G(s) – Nenner:", self._g_den)

        self._c_num = QLineEdit("1 1")
        self._c_num.setPlaceholderText("z.B. 1 1")
        self._c_den = QLineEdit("1")
        self._c_den.setPlaceholderText("z.B. 1")
        input_form.addRow("<b>C(s)</b> – Zähler:", self._c_num)
        input_form.addRow("C(s) – Nenner:", self._c_den)
        main_layout.addWidget(input_group)

        # Flags-Panel
        flags_group = QGroupBox("Topologie-Flags")
        flags_layout = QHBoxLayout(flags_group)

        left_flags = QVBoxLayout()
        self._cb_feedforward = QCheckBox("Vorsteuerung (Feedforward)")
        self._cb_cascade = QCheckBox("Kaskadenregelung")
        self._cb_fsf = QCheckBox("Zustandsrückführung (Full-State)")
        self._cb_mimo = QCheckBox("MIMO-System")
        left_flags.addWidget(self._cb_feedforward)
        left_flags.addWidget(self._cb_cascade)
        left_flags.addWidget(self._cb_fsf)
        left_flags.addWidget(self._cb_mimo)
        flags_layout.addLayout(left_flags)

        right_flags = QVBoxLayout()
        self._cb_hysteresis = QCheckBox("Hysterese")
        hyst_row = QHBoxLayout()
        hyst_row.addWidget(self._cb_hysteresis)
        hyst_label = QLabel("Band:")
        self._sb_hysteresis_band = QDoubleSpinBox()
        self._sb_hysteresis_band.setRange(0.0, 1e6)
        self._sb_hysteresis_band.setDecimals(3)
        self._sb_hysteresis_band.setValue(0.0)
        self._sb_hysteresis_band.setSingleStep(0.1)
        hyst_row.addWidget(hyst_label)
        hyst_row.addWidget(self._sb_hysteresis_band)
        hyst_row.addStretch()
        right_flags.addLayout(hyst_row)

        self._cb_disturbance = QCheckBox("Störgrößen-Kanal")
        right_flags.addWidget(self._cb_disturbance)

        thresh_row = QHBoxLayout()
        thresh_label = QLabel("Schwellwert:")
        self._sb_threshold = QDoubleSpinBox()
        self._sb_threshold.setRange(0.0, 1.0)
        self._sb_threshold.setDecimals(2)
        self._sb_threshold.setValue(0.5)
        self._sb_threshold.setSingleStep(0.05)
        self._sb_threshold.setToolTip("Mindest-Konfidenz für eine Klassenzuweisung")
        thresh_row.addWidget(thresh_label)
        thresh_row.addWidget(self._sb_threshold)
        thresh_row.addStretch()
        right_flags.addLayout(thresh_row)

        flags_layout.addLayout(right_flags)
        main_layout.addWidget(flags_group)

        # Buttons
        btn_row = QHBoxLayout()
        self._classify_btn = QPushButton("Klassifizieren")
        self._classify_btn.setStyleSheet(
            "QPushButton { background-color: #2c3e50; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #34495e; }"
        )
        self._classify_btn.clicked.connect(self._on_classify)
        self._clear_btn = QPushButton("Ergebnis löschen")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_row.addStretch()
        btn_row.addWidget(self._classify_btn)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch()
        main_layout.addLayout(btn_row)

        # Ergebnis-Badge
        badge_row = QHBoxLayout()
        badge_row.addStretch()
        self._badge = _BioBadge()
        badge_row.addWidget(self._badge)
        badge_row.addStretch()
        main_layout.addLayout(badge_row)

        # Subgraph-Kette
        chain_row = QHBoxLayout()
        chain_row.addWidget(QLabel("<b>Subgraph-Kette:</b>"))
        self._chain_label = QLabel("–")
        self._chain_label.setWordWrap(True)
        chain_row.addWidget(self._chain_label)
        chain_row.addStretch()
        main_layout.addLayout(chain_row)

        # Adjazenzmatrix
        self._mat_view = _MatrixView("Adjazenzmatrix")
        main_layout.addWidget(self._mat_view)

        # Erweiterungen / textuelle Zusammenfassung
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumHeight(150)
        self._result_text.setPlaceholderText(
            "Klassifikationsergebnis und Erweiterungsvorschläge erscheinen hier …"
        )
        main_layout.addWidget(self._result_text)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_classify(self) -> None:
        """Liest Eingaben, ruft classify_loop() auf und zeigt das Ergebnis."""
        try:
            g_num = _parse_coeffs(self._g_num.text())
            g_den = _parse_coeffs(self._g_den.text())
            c_num = _parse_coeffs(self._c_num.text())
            c_den = _parse_coeffs(self._c_den.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Eingabefehler", f"Ungültige Koeffizienten:\n{exc}")
            return

        plant = TransferFunction(g_num, g_den, name="G")
        controller = TransferFunction(c_num, c_den, name="C")

        try:
            result = classify_loop(
                plant,
                controller,
                has_feedforward=self._cb_feedforward.isChecked(),
                has_cascade=self._cb_cascade.isChecked(),
                is_full_state_feedback=self._cb_fsf.isChecked(),
                is_mimo=self._cb_mimo.isChecked(),
                has_hysteresis=self._cb_hysteresis.isChecked(),
                hysteresis_band=self._sb_hysteresis_band.value(),
                has_disturbance_channel=self._cb_disturbance.isChecked(),
                threshold=self._sb_threshold.value(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Klassifikationsfehler", str(exc))
            return

        self._last_result = result
        self._display_result(result)
        self.classification_done.emit(result)

    def _on_clear(self) -> None:
        self._badge._set_neutral()
        self._chain_label.setText("–")
        self._mat_view.clear_matrix()
        self._result_text.clear()
        self._last_result = None

    # ------------------------------------------------------------------
    # Ergebnis anzeigen
    # ------------------------------------------------------------------

    def _display_result(self, result: BioClassResult) -> None:
        self._badge.update_class(result.bio_class, result.confidence)

        chain_parts = [c.__name__ if hasattr(c, "__name__") else str(c)
                       for c in result.subgraph_chain]
        self._chain_label.setText(" → ".join(chain_parts) if chain_parts else "–")

        self._mat_view.set_matrix(result.adjacency_matrix)

        extensions = get_extensions(result.bio_class)
        lines: list[str] = [
            f"<b>Klasse:</b> {result.bio_class.label}",
            f"<b>Konfidenz:</b> {result.confidence:.1%}",
            f"<b>Dichte:</b> {result.density:.3f}",
            f"<b>Selbstschleifen:</b> {'ja' if result.has_self_loop else 'nein'}",
            f"<b>Integrator im Regler:</b> {'ja' if result.controller_has_integrator else 'nein'}",
            f"<b>Differenzierer im Regler:</b> {'ja' if result.controller_has_derivative else 'nein'}",
            "",
            f"<b>Zusammenfassung:</b> {result.summary}",
        ]
        if extensions:
            lines += ["", "<b>Erweiterungsvorschläge:</b>"]
            for ext in extensions:
                lines.append(f"&nbsp;&nbsp;• <b>{ext.name}</b>: {ext.description}")
                if ext.code_hint:
                    lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;<i>Hinweis:</i> {ext.code_hint}")
        self._result_text.setHtml("<br>".join(lines))

    # ------------------------------------------------------------------
    # Externe Schnittstelle
    # ------------------------------------------------------------------

    def set_loop(self, plant: TransferFunction, controller: TransferFunction) -> None:
        """Überträgt Strecke und Regler in die Eingabefelder.

        Wird typischerweise vom :class:`~pylabb.gui.main_window.MainWindow`
        aufgerufen, wenn sich das aktive System ändert.

        Parameters
        ----------
        plant : TransferFunction
            Strecke G(s). Die Koeffizienten werden in die Felder
            *G(s) – Zähler* und *G(s) – Nenner* geschrieben.
        controller : TransferFunction
            Regler C(s). Wird in die Felder *C(s) – Zähler/Nenner* übernommen.
        """
        self._g_num.setText(" ".join(str(c) for c in plant.num.real))
        self._g_den.setText(" ".join(str(c) for c in plant.den.real))
        self._c_num.setText(" ".join(str(c) for c in controller.num.real))
        self._c_den.setText(" ".join(str(c) for c in controller.den.real))

    @property
    def last_result(self) -> Optional[BioClassResult]:
        """Letztes vollständiges Klassifikationsergebnis.

        Returns
        -------
        BioClassResult | None
            Das zuletzt von :func:`~pylabb.control.bio_classify.classify_loop`
            zurückgegebene Ergebnis, oder ``None`` falls noch keine Klassifikation
            durchgeführt oder das Ergebnis mit *Ergebnis löschen* zurückgesetzt wurde.
        """
        return self._last_result

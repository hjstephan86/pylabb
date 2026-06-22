"""
pylabb.gui.widgets.analysis_widget
===================================
Widget für die Stabilitätsanalyse: Zeigt Kenngrößen, Pol-Nullstellen,
Stabilitätsränder und Hinweis-Indikatoren an.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QTextEdit, QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.stability import StabilityInfo, analyze


class StabilityBadge(QLabel):
    """Einfaches farbiges Status-Label (stabil / instabil / unbekannt).

    Das Label passt Hintergrundfarbe und Text automatisch an den
    Stabilitätszustand an (grün = stabil, rot = instabil).
    """

    def set_stable(self, stable: bool) -> None:
        """Setzt den angezeigten Stabilitätszustand.

        Parameters
        ----------
        stable : bool
            ``True``  → grünes Badge  "  STABIL  "
            ``False`` → rotes  Badge  "  INSTABIL  "
        """
        if stable:
            self.setText("  STABIL  ")
            self.setStyleSheet(
                "background-color:#27ae60;color:white;font-weight:bold;"
                "border-radius:4px;padding:4px;"
            )
        else:
            self.setText("  INSTABIL  ")
            self.setStyleSheet(
                "background-color:#c0392b;color:white;font-weight:bold;"
                "border-radius:4px;padding:4px;"
            )


class AnalysisWidget(QWidget):
    """Vollständiges Stabilitätsanalyse-Panel.

    Zeigt Stabilitätskenngrößen (Amplitudenrand, Phasenrand,
    Pol-/Nullstellenverteilung) für ein übergebenes System
    :class:`~pylabb.core.transfer_function.TransferFunction` an.

    Parameters
    ----------
    parent : QWidget, optional
        Übergeordnetes Widget.

    Attributes
    ----------
    last_info : StabilityInfo | None
        Letztes Analyseergebnis oder ``None``.

    Signals
    -------
    analysis_done(StabilityInfo) : Nach jedem Analysedurchlauf.
    """

    analysis_done = pyqtSignal(object)  # StabilityInfo

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_info: StabilityInfo | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI-Aufbau
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Kopfzeile mit Status-Badge
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Stabilitätsanalyse</b>"))
        header.addStretch()
        self._badge = StabilityBadge("  –  ")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self._badge)
        main_layout.addLayout(header)

        # Splitter: Tabelle + Textausgabe
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Kennwert-Tabelle
        self._table = QTableWidget(7, 2)
        self._table.setHorizontalHeaderLabels(["Kenngröße", "Wert"])
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        splitter.addWidget(self._table)

        # Pol/Nullstellen-Textausgabe
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Courier", 9))
        self._text.setMaximumHeight(180)
        splitter.addWidget(self._text)

        main_layout.addWidget(splitter)

        # Analyse-Button
        btn = QPushButton("Jetzt analysieren")
        btn.clicked.connect(self._on_analyze_clicked)
        main_layout.addWidget(btn)

        # Initialen Zustand setzen
        self._populate_table(None)

    def _populate_table(self, info: StabilityInfo | None) -> None:
        rows = [
            ("Stabil", str(info.is_stable) if info else "–"),
            ("Instabile Pole (RHP)", str(info.rhp_poles) if info else "–"),
            ("Amplitudenrand [dB]", f"{info.gain_margin_dB:.2f}" if info else "–"),
            ("Phasenrand [°]", f"{info.phase_margin_deg:.2f}" if info else "–"),
            ("Durchtrittsfrequenz [rad/s]", f"{info.gain_crossover_freq:.4g}" if info else "–"),
            ("Phasenumkehrfrequenz [rad/s]", f"{info.phase_crossover_freq:.4g}" if info else "–"),
            ("Systemordnung", str(len(info.poles)) if info else "–"),
        ]
        self._table.setRowCount(len(rows))
        for i, (key, val) in enumerate(rows):
            k_item = QTableWidgetItem(key)
            v_item = QTableWidgetItem(val)
            if info and key == "Stabil":
                v_item.setForeground(
                    QColor("#27ae60") if info.is_stable else QColor("#c0392b")
                )
            self._table.setItem(i, 0, k_item)
            self._table.setItem(i, 1, v_item)
        self._table.resizeColumnsToContents()

    def _on_analyze_clicked(self) -> None:
        """Platzhalter – wird von MainWindow überschrieben."""
        self._text.setPlainText("Kein System geladen. Bitte zuerst eine Strecke definieren.")

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def run_analysis(self, G: TransferFunction) -> StabilityInfo:
        """Führt die Stabilitätsanalyse durch und aktualisiert alle UI-Elemente.

        Ruft :func:`pylabb.control.stability.analyze` auf, befüllt die
        Kennwert-Tabelle, setzt das :class:`StabilityBadge` und gibt das
        Ergebnis über :attr:`analysis_done` aus.

        Parameters
        ----------
        G : TransferFunction
            Zu analysierendes Übertragungsfunktions-Objekt.

        Returns
        -------
        StabilityInfo
            Vollständiges Analyseergebnis (Pole, Ränder, Kreuzfrequenzen).
        """
        info = analyze(G)
        self._last_info = info
        self._badge.set_stable(info.is_stable)
        self._populate_table(info)

        lines = [str(info), ""]
        self._text.setPlainText("\n".join(lines))
        self.analysis_done.emit(info)
        return info

    @property
    def last_info(self) -> StabilityInfo | None:
        """Letztes vollständiges Analyseergebnis.

        Returns
        -------
        StabilityInfo | None
            Das von :meth:`run_analysis` zuletzt zurückgegebene Ergebnis,
            oder ``None`` falls noch keine Analyse durchgeführt wurde.
        """
        return self._last_info

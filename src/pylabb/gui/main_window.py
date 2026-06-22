"""
pylabb.gui.main_window
======================
Haupt-GUI-Fenster für das pylab-Framework.

Layout
------
┌─────────────────────────────────────────────────────────┐
│ Menüleiste + Werkzeugleiste                             │
├──────────────┬──────────────────────────────────────────┤
│              │  MultiPlotWidget (Tabs)                  │
│  System-     │   • Bode                                 │
│  Editor      │   • Nyquist                              │
│  (Dock)      │   • Sprungantwort                        │
│              │   • Pol-Nullstellen                      │
├──────────────┼──────────────────────────────────────────┤
│  Analyse     │  Codegen-Widget                          │
│  (Dock)      │                                          │
└──────────────┴──────────────────────────────────────────┘
│ Statusleiste                                            │
└─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import sys
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget,
    QSplitter, QVBoxLayout, QStatusBar, QMenuBar,
    QToolBar, QMessageBox, QTabWidget, QLabel, QScrollArea,
)
from PyQt6.QtCore import Qt, QSize, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QFont

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.pid import PIDController
from pylabb.gui.widgets.plot_widget import MultiPlotWidget, PlotWidget
from pylabb.gui.widgets.system_editor import SystemEditorPanel
from pylabb.gui.widgets.analysis_widget import AnalysisWidget
from pylabb.gui.widgets.codegen_widget import CodegenWidget
from pylabb.gui.widgets.verification_widget import VerificationWidget
from pylabb.gui.widgets.bio_classify_widget import BioClassifyWidget
from pylabb import __version__ as _pylabb_version, __author__ as _pylabb_author


def _scrolled(widget: QWidget) -> QScrollArea:
    """Bettet ein Widget in eine QScrollArea ein."""
    sa = QScrollArea()
    sa.setWidget(widget)
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    sa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return sa


class MainWindow(QMainWindow):
    """PyLab-Hauptfenster."""

    APP_TITLE = f"PyLab  {_pylabb_version}  –  Regelungstechnik & MicroPython-Codegen"

    def __init__(self) -> None:
        super().__init__()
        self._plant: Optional[TransferFunction] = None
        self._controller: Optional[PIDController] = None

        self.setWindowTitle(self.APP_TITLE)

        self._build_menus()
        self._build_toolbars()
        self._build_central()
        self._build_docks()
        self._build_statusbar()
        self._connect_signals()

        # Standardsystem laden
        self._load_default_system()

        self.showMaximized()

    # ------------------------------------------------------------------
    # UI-Aufbau
    # ------------------------------------------------------------------

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # Datei-Menü
        file_menu = mb.addMenu("&Datei")
        self._act_new = QAction("&Neues Projekt", self, shortcut="Ctrl+N",
                                triggered=self._on_new_project)
        self._act_exit = QAction("&Beenden", self, shortcut="Ctrl+Q",
                                 triggered=self.close)
        file_menu.addAction(self._act_new)
        file_menu.addSeparator()
        file_menu.addAction(self._act_exit)

        # Analyse-Menü
        analysis_menu = mb.addMenu("&Analyse")
        self._act_analyze = QAction("Stabilitätsanalyse ausführen", self,
                                    shortcut="F5", triggered=self._on_run_analysis)
        self._act_bode = QAction("Bode-Diagramm", self, triggered=self._on_plot_bode)
        self._act_nyquist = QAction("Nyquist-Ortskurve", self, triggered=self._on_plot_nyquist)
        self._act_rlocus = QAction("Wurzelortskurve", self, triggered=self._on_plot_rlocus)
        self._act_step = QAction("Sprungantwort", self, triggered=self._on_plot_step)
        self._act_sim = QAction("Geschlossener Regelkreis simulieren", self,
                                triggered=self._on_simulate_closed_loop)
        analysis_menu.addActions([
            self._act_analyze, self._act_bode, self._act_nyquist,
            self._act_rlocus, self._act_step,
        ])
        analysis_menu.addSeparator()
        analysis_menu.addAction(self._act_sim)

        # Codegen-Menü
        codegen_menu = mb.addMenu("&Codegen")
        self._act_gen = QAction("MicroPython-Code generieren", self,
                                shortcut="F6", triggered=self._on_codegen)
        self._act_export = QAction("Projektbundle exportieren …", self,
                                   triggered=self._on_export)
        codegen_menu.addActions([self._act_gen, self._act_export])

        # Verifikation-Menü
        verify_menu = mb.addMenu("&Verifikation")
        self._act_verify = QAction(
            "Regelkreis-Äquivalenz prüfen", self,
            shortcut="F7",
            triggered=self._on_verify_loops,
        )
        self._act_load_A = QAction(
            "Aktuellen Regelkreis als A übernehmen", self,
            triggered=self._on_load_current_as_A,
        )
        verify_menu.addAction(self._act_verify)
        verify_menu.addSeparator()
        verify_menu.addAction(self._act_load_A)

        # Biologische Klassifikation-Menü
        bio_menu = mb.addMenu("&Bio-Klassifikation")
        self._act_bio_classify = QAction(
            "Biologische Klasse bestimmen", self,
            shortcut="F8",
            triggered=self._on_bio_classify,
        )
        self._act_load_current_loop = QAction(
            "Aktuellen Regelkreis übernehmen", self,
            triggered=self._on_load_current_loop,
        )
        bio_menu.addAction(self._act_bio_classify)
        bio_menu.addSeparator()
        bio_menu.addAction(self._act_load_current_loop)

        # Hilfe-Menü
        help_menu = mb.addMenu("&Hilfe")
        self._act_about = QAction("&Über PyLab", self, triggered=self._on_about)
        help_menu.addAction(self._act_about)

    def _build_toolbars(self) -> None:
        tb = QToolBar("Hauptwerkzeuge")
        tb.setIconSize(QSize(24, 24))
        self.addToolBar(tb)

        tb.addAction(self._act_analyze)
        tb.addSeparator()
        tb.addAction(self._act_bode)
        tb.addAction(self._act_nyquist)
        tb.addAction(self._act_rlocus)
        tb.addAction(self._act_step)
        tb.addSeparator()
        tb.addAction(self._act_sim)
        tb.addSeparator()
        tb.addAction(self._act_gen)
        tb.addSeparator()
        tb.addAction(self._act_verify)
        tb.addAction(self._act_bio_classify)

    def _build_central(self) -> None:
        """Zentraler Bereich: Tabs mit Plot-Widgets + Codegen."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Plot-Tabs
        self._plots = MultiPlotWidget(tabs=[
            "Bode",
            "Nyquist",
            "Sprungantwort",
            "Pol-Nullstellen",
            "Simulation",
            "Spektrum",
        ])
        main_splitter.addWidget(self._plots)

        # Codegen-Panel (eingeklappt unten)
        self._codegen_widget = CodegenWidget()
        codegen_dock_inner = QTabWidget()
        codegen_dock_inner.addTab(self._codegen_widget, "MicroPython-Codegen")
        main_splitter.addWidget(codegen_dock_inner)

        main_splitter.setSizes([600, 300])
        layout.addWidget(main_splitter)

    def _build_docks(self) -> None:
        # Linkes Dock: System-Editor
        self._system_panel = SystemEditorPanel()
        dock_sys = QDockWidget("System-Editor", self)
        dock_sys.setWidget(_scrolled(self._system_panel))
        dock_sys.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_sys)

        # Linkes Dock 2: Stabilitäts-Analyse
        self._analysis_widget = AnalysisWidget()
        dock_analysis = QDockWidget("Analyse", self)
        dock_analysis.setWidget(_scrolled(self._analysis_widget))
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_analysis)

        # Rechtes Dock: Regelkreis-Verifikation
        self._verification_widget = VerificationWidget()
        dock_verify = QDockWidget("Regelkreis-Verifikation", self)
        dock_verify.setWidget(_scrolled(self._verification_widget))
        dock_verify.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_verify)
        self._dock_verify = dock_verify

        # Rechtes Dock 2: Biologische Klassifikation
        self._bio_classify_widget = BioClassifyWidget()
        dock_bio = QDockWidget("Bio-Klassifikation", self)
        dock_bio.setWidget(_scrolled(self._bio_classify_widget))
        dock_bio.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_bio)
        self._dock_bio = dock_bio

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_label = QLabel("Bereit")
        sb.addWidget(self._status_label)

    def _connect_signals(self) -> None:
        self._system_panel.plant_changed.connect(self._on_plant_changed)
        self._system_panel.controller_changed.connect(self._on_controller_changed)
        self._analysis_widget.analysis_done.connect(
            lambda info: self._status(f"Analyse: {'Stabil' if info.is_stable else 'Instabil'}"
                                       f" | GM={info.gain_margin_dB:.1f}dB"
                                       f" | PM={info.phase_margin_deg:.1f}°")
        )
        self._verification_widget.verification_done.connect(
            lambda r: self._status(f"Verifikation: {r.decision} – {r.summary}")
        )
        self._bio_classify_widget.classification_done.connect(
            lambda r: self._status(
                f"Bio-Klasse: {r.bio_class.label} (Konfidenz: {r.confidence:.0%})"
            )
        )
        # Codegen-Analyse-Button im AnalysisWidget überschreiben
        self._analysis_widget._on_analyze_clicked = self._on_run_analysis  # type: ignore

    # ------------------------------------------------------------------
    # Standardsystem
    # ------------------------------------------------------------------

    def _load_default_system(self) -> None:
        """Lädt ein PT2-Beispielsystem beim Start."""
        from pylabb.core.transfer_function import TransferFunction
        G = TransferFunction([1], [1, 1.4, 1], name="G_PT2")
        self._plant = G
        pid = PIDController(Kp=2.0, Ti=1.0, Td=0.1)
        self._controller = pid
        self._system_panel.tf_editor.set_tf(G)
        self._codegen_widget.set_pid(pid)
        self._codegen_widget.set_plant_tf(G)
        self._on_run_analysis()
        self._on_plot_bode()
        self._status("Standardsystem geladen: G(s) = 1 / (s² + 1.4s + 1)")

    # ------------------------------------------------------------------
    # Signal-Handler: System-Zustand
    # ------------------------------------------------------------------

    @pyqtSlot(object)
    def _on_plant_changed(self, tf: TransferFunction) -> None:
        self._plant = tf
        self._codegen_widget.set_plant_tf(tf)
        self._status(f"Strecke geändert: {tf}")
        self._sync_verification_loop_A()

    @pyqtSlot(object)
    def _on_controller_changed(self, pid: PIDController) -> None:
        self._controller = pid
        self._codegen_widget.set_pid(pid)
        self._status(f"Regler geändert: {pid}")
        self._sync_verification_loop_A()

    # ------------------------------------------------------------------
    # Analyse-Aktionen
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_run_analysis(self) -> None:
        if not self._check_plant():
            return
        try:
            self._analysis_widget.run_analysis(self._plant)
        except Exception as ex:
            self._error("Analysefehler", str(ex))

    @pyqtSlot()
    def _on_plot_bode(self) -> None:
        if not self._check_plant():
            return
        try:
            pw = self._plots.get_plot("Bode")
            pw.clear()
            from pylabb.visualization.bode import plot_bode
            G_list = [self._plant]
            if self._controller:
                C_tf = self._controller.transfer_function()
                from pylabb.control.design import closed_loop, series
                L = series(C_tf, self._plant)
                L.name = "L=C·G"
                G_list.append(L)
            plot_bode(*G_list, fig=pw.fig, title="Bode-Diagramm")
            pw.redraw()
            self._status("Bode-Diagramm aktualisiert")
        except Exception as ex:
            self._error("Bode-Fehler", str(ex))

    @pyqtSlot()
    def _on_plot_nyquist(self) -> None:
        if not self._check_plant():
            return
        try:
            pw = self._plots.get_plot("Nyquist")
            pw.clear()
            from pylabb.visualization.nyquist import plot_nyquist
            plot_nyquist(self._plant, fig=pw.fig, title="Nyquist-Ortskurve")
            pw.redraw()
            self._status("Nyquist-Ortskurve aktualisiert")
        except Exception as ex:
            self._error("Nyquist-Fehler", str(ex))

    @pyqtSlot()
    def _on_plot_rlocus(self) -> None:
        if not self._check_plant():
            return
        try:
            pw = self._plots.get_plot("Pol-Nullstellen")
            pw.clear()
            from pylabb.visualization.rlocus import plot_root_locus
            from pylabb.visualization.time_plots import plot_pole_zero
            plot_root_locus(self._plant, fig=pw.fig, title="Wurzelortskurve")
            pw.redraw()
            self._status("Wurzelortskurve aktualisiert")
        except Exception as ex:
            self._error("WOK-Fehler", str(ex))

    @pyqtSlot()
    def _on_plot_step(self) -> None:
        if not self._check_plant():
            return
        try:
            pw = self._plots.get_plot("Sprungantwort")
            pw.clear()
            from pylabb.visualization.time_plots import plot_step_response
            systems = [self._plant]
            if self._controller:
                from pylabb.control.design import closed_loop
                C_tf = self._controller.transfer_function()
                T = closed_loop(self._plant, C_tf)
                T.name = "T(s) ges. Kreis"
                systems.append(T)
            plot_step_response(*systems, fig=pw.fig, title="Sprungantwort")
            pw.redraw()
            self._status("Sprungantwort aktualisiert")
        except Exception as ex:
            self._error("Sprungantwort-Fehler", str(ex))

    @pyqtSlot()
    def _on_simulate_closed_loop(self) -> None:
        if not self._check_plant():
            return
        if self._controller is None:
            self._warn("Kein Regler definiert.", "Bitte zuerst einen PID-Regler eingeben.")
            return
        try:
            from pylabb.core.signals import step_signal
            from pylabb.simulation.time_domain import ClosedLoopSimulator
            from pylabb.visualization.time_plots import plot_simulation

            C_tf = self._controller.transfer_function()
            sim = ClosedLoopSimulator(self._plant, C_tf, name="Regelkreis")
            w = step_signal(t_end=20.0, dt=0.01)
            result = sim.run(w)

            pw = self._plots.get_plot("Simulation")
            pw.clear()
            plot_simulation(result, fig=pw.fig, title="Regelkreis-Simulation")
            pw.redraw()
            s = result.summary()
            self._status(
                f"Simulation: ts={s['Einschwingzeit [s]']:.3g}s, "
                f"Ü={s['Überschwingen [%]']:.2g}%, "
                f"IAE={s['IAE']:.3g}"
            )
        except Exception as ex:
            self._error("Simulationsfehler", str(ex))

    # ------------------------------------------------------------------
    # Codegen-Aktionen
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_codegen(self) -> None:
        self._codegen_widget._on_generate()

    @pyqtSlot()
    def _on_export(self) -> None:
        self._codegen_widget._on_export_project()

    # ------------------------------------------------------------------
    # Sonstige Aktionen
    # ------------------------------------------------------------------

    def _sync_verification_loop_A(self) -> None:
        """Überträgt den aktuellen Regelkreis in Slot A des Verifikations-Widgets."""
        if self._plant is None:
            return
        ctrl_tf = (
            self._controller.transfer_function()
            if self._controller is not None
            else TransferFunction([1.0], [1.0], name="C")
        )
        self._verification_widget.set_loop_A(self._plant, ctrl_tf)

    def _sync_bio_classify_loop(self) -> None:
        """Überträgt den aktuellen Regelkreis in das Bio-Klassifikations-Widget."""
        if self._plant is None:
            return
        ctrl_tf = (
            self._controller.transfer_function()
            if self._controller is not None
            else TransferFunction([1.0], [1.0], name="C")
        )
        self._bio_classify_widget.set_loop(self._plant, ctrl_tf)

    @pyqtSlot()
    def _on_verify_loops(self) -> None:
        """Öffnet das Verifikations-Dock und löst die Prüfung aus."""
        self._dock_verify.show()
        self._dock_verify.raise_()

    @pyqtSlot()
    def _on_bio_classify(self) -> None:
        """Öffnet das Bio-Klassifikations-Dock."""
        self._dock_bio.show()
        self._dock_bio.raise_()

    @pyqtSlot()
    def _on_load_current_loop(self) -> None:
        """Überträgt den aktuellen Regelkreis in das Bio-Klassifikations-Widget."""
        self._sync_bio_classify_loop()
        self._dock_bio.show()
        self._dock_bio.raise_()
        self._status("Aktueller Regelkreis in Bio-Klassifikation übernommen.")

    @pyqtSlot()
    def _on_load_current_as_A(self) -> None:
        """Kopiert den aktuellen Regelkreis in Slot A des Verifikations-Widgets."""
        self._sync_verification_loop_A()
        self._dock_verify.show()
        self._dock_verify.raise_()
        self._sync_bio_classify_loop()
        self._status("Aktueller Regelkreis als Regelkreis A übernommen.")

    @pyqtSlot()
    def _on_new_project(self) -> None:
        self._plant = None
        self._controller = None
        self._plots.clear_all()
        self._analysis_widget._populate_table(None)
        self._analysis_widget._badge.setText("  –  ")
        self._status("Neues Projekt erstellt.")


    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "Über PyLab",
            f"<b>PyLab {_pylabb_version}</b><br/><br/>"
            "Umfassendes Framework für mathematische Berechnungen,<br/>"
            "Regelungstechnik und MicroPython-Codegenerierung.<br/><br/>"
            f"Autor: {_pylabb_author}<br/><br/>"
            "Verwendete Bibliotheken:<br/>"
            "NumPy · SciPy · Matplotlib · PyQt6",
        )

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _check_plant(self) -> bool:
        if self._plant is None:
            self._warn("Kein System", "Bitte zuerst eine Strecke G(s) definieren.")
            return False
        return True

    def _status(self, msg: str) -> None:
        self._status_label.setText(msg)

    def _warn(self, title: str, msg: str) -> None:
        QMessageBox.warning(self, title, msg)

    def _error(self, title: str, msg: str) -> None:
        QMessageBox.critical(self, title, msg)


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

def main() -> None:
    """Startet die PyLab-GUI (Entry Point für pyproject.toml-Script)."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("PyLab")
    app.setOrganizationName("pylab")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

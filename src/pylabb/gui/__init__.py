"""
pylabb.gui – PyQt6-basierte grafische Benutzeroberfläche
=========================================================

Dieses Paket enthält das vollständige PyQt6-GUI-Frontend für PyLab:

Hauptfenster
------------
:class:`MainWindow` – Zentrales QMainWindow mit Menüleiste, Werkzeugleiste,
Dock-Widgets und Plot-Tabs.

Dock-Widgets (in ``gui.widgets``)
----------------------------------
* :class:`~pylabb.gui.widgets.system_editor.SystemEditorPanel` –
  Strecke G(s) und PID-Regler C(s) bearbeiten
* :class:`~pylabb.gui.widgets.analysis_widget.AnalysisWidget` –
  Stabilitätsanalyse und Kenngrößen
* :class:`~pylabb.gui.widgets.verification_widget.VerificationWidget` –
  Subgraph-Äquivalenzprüfung zweier Regelkreise
* :class:`~pylabb.gui.widgets.bio_classify_widget.BioClassifyWidget` –
  Biologische Äquivalenzklassen-Klassifikation
* :class:`~pylabb.gui.widgets.codegen_widget.CodegenWidget` –
  MicroPython-Codegenerierung

Plot-Panel (in ``gui.widgets``)
--------------------------------
* :class:`~pylabb.gui.widgets.plot_widget.MultiPlotWidget` –
  Tabbed Plots (Bode, Nyquist, Sprungantwort, Pol-Nullstellen, …)

Einstiegspunkt
--------------
:func:`main` startet die Applikation (wird in ``pyproject.toml`` als
Script-Einstiegspunkt registriert).
"""

from .main_window import MainWindow, main
from . import widgets

__all__ = ["MainWindow", "main", "widgets"]

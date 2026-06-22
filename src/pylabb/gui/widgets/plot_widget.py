"""
pylabb.gui.widgets.plot_widget
==============================
Matplotlib-Canvas in einem PyQt6-Widget – wiederverwendbar für alle
Diagrammtypen (Bode, Nyquist, Zeitgang …).

Enthält:
  ``MplCanvas``           – QWidget mit eingebettetem Matplotlib-Figure.
  ``PlotWidget``          – Erweiterter Widget mit Toolbar und Zoom/Pan.
  ``MultiPlotWidget``     – Tabbed-Widget mit mehreren Plots.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")  # Qt-Backend vor anderen Imports setzen

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import matplotlib.pyplot as plt
import matplotlib.figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTabWidget, QSizePolicy, QLabel, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal


# ---------------------------------------------------------------------------
# Einfacher Canvas
# ---------------------------------------------------------------------------

class MplCanvas(FigureCanvas):
    """Matplotlib-Figure als QWidget."""

    def __init__(
        self,
        parent: QWidget | None = None,
        figsize: tuple[float, float] = (8, 5),
        dpi: int = 100,
    ) -> None:
        fig = matplotlib.figure.Figure(figsize=figsize, dpi=dpi, tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

    @property
    def fig(self) -> matplotlib.figure.Figure:
        return self.figure

    def clear(self) -> None:
        """Löscht alle Axes."""
        self.figure.clear()
        self.draw()

    def get_or_add_axes(self, rows: int = 1, cols: int = 1, idx: int = 1):
        axes = self.figure.get_axes()
        if not axes:
            return self.figure.add_subplot(rows, cols, idx)
        return axes[min(idx - 1, len(axes) - 1)]


# ---------------------------------------------------------------------------
# PlotWidget mit Toolbar
# ---------------------------------------------------------------------------

class PlotWidget(QWidget):
    """Widget bestehend aus Matplotlib-Canvas + Navigationstoolbar.

    Signals
    -------
    plot_updated : Wird nach jedem Neuzeichnen emittiert.
    """

    plot_updated = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        figsize: tuple[float, float] = (9, 6),
        title: str = "",
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._canvas = MplCanvas(self, figsize=figsize)
        self._toolbar = NavigationToolbar(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    @property
    def canvas(self) -> MplCanvas:
        return self._canvas

    @property
    def fig(self) -> matplotlib.figure.Figure:
        return self._canvas.figure

    def clear(self) -> None:
        """Löscht das Figure."""
        self._canvas.figure.clear()
        self._canvas.draw()
        self.plot_updated.emit()

    def redraw(self) -> None:
        """Neuzeichnen nach externen Änderungen am Figure."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._canvas.figure.tight_layout()
        self._canvas.draw()
        self.plot_updated.emit()

    def save_dialog(self) -> None:
        """Öffnet einen Speicher-Dialog für den aktuellen Plot."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Plot speichern",
            filter="PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if path:
            self._canvas.figure.savefig(path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Multi-Plot (Tabs)
# ---------------------------------------------------------------------------

class MultiPlotWidget(QWidget):
    """Tabellarisches Widget mit mehreren benannten PlotWidgets.

    Beispiel::

        mpw = MultiPlotWidget(tabs=["Bode", "Nyquist", "Sprungantwort"])
        bode_fig = mpw.get_fig("Bode")
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        tabs: list[str] | None = None,
        figsize: tuple[float, float] = (9, 6),
    ) -> None:
        super().__init__(parent)
        self._tab_widget = QTabWidget(self)
        self._plots: dict[str, PlotWidget] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tab_widget)

        for name in (tabs or ["Plot"]):
            self.add_tab(name, figsize=figsize)

    def add_tab(
        self,
        name: str,
        figsize: tuple[float, float] = (9, 6),
    ) -> PlotWidget:
        """Fügt einen neuen Tab hinzu und gibt den PlotWidget zurück."""
        pw = PlotWidget(figsize=figsize, title=name)
        self._tab_widget.addTab(pw, name)
        self._plots[name] = pw
        return pw

    def get_plot(self, name: str) -> PlotWidget:
        """Gibt den PlotWidget für den benannten Tab zurück."""
        return self._plots[name]

    def get_fig(self, name: str) -> matplotlib.figure.Figure:
        """Gibt das Figure für den benannten Tab zurück."""
        return self._plots[name].fig

    def clear_all(self) -> None:
        """Löscht alle Plots."""
        for pw in self._plots.values():
            pw.clear()

    def redraw_all(self) -> None:
        """Zeichnet alle Plots neu."""
        for pw in self._plots.values():
            pw.redraw()

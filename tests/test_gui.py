"""
Tests für alle pylabb.gui-Module.

Läuft ohne physisches Display dank QT_QPA_PLATFORM=offscreen
(wird in conftest.py vor jedem Qt-Import gesetzt).
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

# -----------------------------------------------------------------------
# Imports – triggern auch gui/__init__.py und gui/widgets/__init__.py
# -----------------------------------------------------------------------
import pylabb.gui  # noqa: F401 – covers gui/__init__.py lines 5-7
import pylabb.gui.widgets  # noqa: F401 – covers gui/widgets/__init__.py lines 2-7

from pylabb.gui.widgets.plot_widget import MplCanvas, PlotWidget, MultiPlotWidget
from pylabb.gui.widgets.analysis_widget import StabilityBadge, AnalysisWidget
from pylabb.gui.widgets.codegen_widget import _PythonHighlighter, CodegenWidget
from pylabb.gui.widgets.system_editor import (
    _parse_coeffs,
    TransferFunctionEditor,
    PIDEditor,
    SystemEditorPanel,
)
from pylabb.gui.main_window import MainWindow

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.pid import PIDController


# =========================================================================
# Helpers: ensure QApplication exists (conftest.py provides qapp fixture,
# but some tests just need it to exist)
# =========================================================================

@pytest.fixture(autouse=True)
def _ensure_app(qapp):
    """Make the session QApplication available to every test."""
    return qapp


# =========================================================================
# gui/__init__.py  &  gui/widgets/__init__.py
# =========================================================================

class TestGUIPackageImports:
    def test_main_window_exported(self):
        from pylabb.gui import MainWindow as MW
        assert MW is MainWindow

    def test_main_exported(self):
        from pylabb.gui import main
        assert callable(main)

    def test_widgets_exports(self):
        from pylabb.gui.widgets import (
            MplCanvas, PlotWidget, MultiPlotWidget,
            TransferFunctionEditor, PIDEditor, SystemEditorPanel,
            AnalysisWidget, CodegenWidget,
        )
        assert all(cls is not None for cls in [
            MplCanvas, PlotWidget, MultiPlotWidget,
            TransferFunctionEditor, PIDEditor, SystemEditorPanel,
            AnalysisWidget, CodegenWidget,
        ])


# =========================================================================
# plot_widget.py
# =========================================================================

class TestMplCanvas:
    def test_construction(self):
        c = MplCanvas()
        assert c.fig is not None

    def test_fig_property(self):
        c = MplCanvas(figsize=(6, 4))
        assert c.fig is c.figure

    def test_clear(self):
        c = MplCanvas()
        c.fig.add_subplot(1, 1, 1)
        c.clear()
        assert len(c.fig.get_axes()) == 0

    def test_get_or_add_axes_creates(self):
        c = MplCanvas()
        ax = c.get_or_add_axes()
        assert ax is not None

    def test_get_or_add_axes_existing(self):
        c = MplCanvas()
        ax1 = c.get_or_add_axes()
        ax2 = c.get_or_add_axes()
        assert ax1 is ax2


class TestPlotWidget:
    def test_construction(self):
        pw = PlotWidget()
        assert pw.canvas is not None

    def test_canvas_property(self):
        pw = PlotWidget()
        assert isinstance(pw.canvas, MplCanvas)

    def test_fig_property(self):
        pw = PlotWidget()
        import matplotlib.figure
        assert isinstance(pw.fig, matplotlib.figure.Figure)

    def test_clear(self):
        pw = PlotWidget()
        pw.fig.add_subplot(1, 1, 1)
        pw.clear()
        assert len(pw.fig.get_axes()) == 0

    def test_redraw(self):
        pw = PlotWidget()
        pw.fig.add_subplot(1, 1, 1)
        # Should not raise
        pw.redraw()

    def test_save_dialog_no_path(self):
        """save_dialog with empty path should be a no-op."""
        pw = PlotWidget()
        with patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName",
                   return_value=("", "")):
            pw.save_dialog()  # should not raise

    def test_save_dialog_with_path(self, tmp_path):
        """save_dialog with a valid path saves the figure."""
        pw = PlotWidget()
        dest = str(tmp_path / "out.png")
        with patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName",
                   return_value=(dest, "PNG (*.png)")):
            pw.save_dialog()
        assert os.path.exists(dest)


class TestMultiPlotWidget:
    def test_construction_default(self):
        mpw = MultiPlotWidget()
        assert mpw is not None

    def test_construction_with_tabs(self):
        mpw = MultiPlotWidget(tabs=["A", "B", "C"])
        assert mpw.get_plot("A") is not None

    def test_add_tab(self):
        mpw = MultiPlotWidget(tabs=[])
        pw = mpw.add_tab("New")
        assert isinstance(pw, PlotWidget)

    def test_get_fig(self):
        mpw = MultiPlotWidget(tabs=["X"])
        import matplotlib.figure
        assert isinstance(mpw.get_fig("X"), matplotlib.figure.Figure)

    def test_clear_all(self):
        mpw = MultiPlotWidget(tabs=["A", "B"])
        mpw.clear_all()  # should not raise

    def test_redraw_all(self):
        mpw = MultiPlotWidget(tabs=["A"])
        mpw.get_fig("A").add_subplot(1, 1, 1)
        mpw.redraw_all()  # should not raise


# =========================================================================
# analysis_widget.py
# =========================================================================

class TestStabilityBadge:
    def test_stable(self):
        badge = StabilityBadge()
        badge.set_stable(True)
        assert "STABIL" in badge.text()
        assert "c0392b" not in badge.styleSheet()

    def test_unstable(self):
        badge = StabilityBadge()
        badge.set_stable(False)
        assert "INSTABIL" in badge.text()
        assert "c0392b" in badge.styleSheet()


class TestAnalysisWidget:
    def test_construction(self):
        aw = AnalysisWidget()
        assert aw._last_info is None

    def test_last_info_initially_none(self):
        aw = AnalysisWidget()
        assert aw.last_info is None

    def test_populate_table_none(self):
        aw = AnalysisWidget()
        aw._populate_table(None)
        # Row 0 should have "–"
        assert aw._table.item(0, 1).text() == "–"

    def test_on_analyze_clicked_no_system(self):
        """Default handler sets placeholder text."""
        aw = AnalysisWidget()
        aw._on_analyze_clicked()
        assert "Kein System" in aw._text.toPlainText()

    def test_run_analysis_stable(self):
        aw = AnalysisWidget()
        G = TransferFunction([1], [1, 2, 1])
        info = aw.run_analysis(G)
        assert info.is_stable
        assert aw.last_info is info
        assert "STABIL" in aw._badge.text()

    def test_run_analysis_unstable(self):
        """System with RHP pole → badge shows instabil."""
        aw = AnalysisWidget()
        G = TransferFunction([1], [1, -1])   # pole at +1
        info = aw.run_analysis(G)
        assert not info.is_stable
        assert "INSTABIL" in aw._badge.text()

    def test_run_analysis_updates_table(self):
        aw = AnalysisWidget()
        G = TransferFunction([1], [1, 1])
        aw.run_analysis(G)
        assert aw._table.item(0, 1).text() != "–"

    def test_analysis_done_signal(self):
        aw = AnalysisWidget()
        received = []
        aw.analysis_done.connect(lambda info: received.append(info))
        G = TransferFunction([1], [1, 2, 1])
        aw.run_analysis(G)
        assert len(received) == 1


# =========================================================================
# codegen_widget.py
# =========================================================================

class TestPythonHighlighter:
    def _make_highlighter(self):
        from PyQt6.QtWidgets import QPlainTextEdit
        edit = QPlainTextEdit()
        hl = _PythonHighlighter(edit.document())
        return hl, edit

    def test_construction(self):
        hl, _ = self._make_highlighter()
        assert hl is not None

    def test_highlight_comment(self):
        hl, edit = self._make_highlighter()
        edit.setPlainText("x = 1  # this is a comment")
        # Just verify no crash

    def test_highlight_keyword(self):
        hl, edit = self._make_highlighter()
        edit.setPlainText("def my_func(self):\n    return True")

    def test_highlight_string(self):
        hl, edit = self._make_highlighter()
        edit.setPlainText("s = 'hello' + \"world\"")

    def test_highlight_number(self):
        hl, edit = self._make_highlighter()
        edit.setPlainText("x = 3.14e-5 + 42")


class TestCodegenWidget:
    def test_construction(self):
        cw = CodegenWidget()
        assert cw is not None

    def test_set_pid(self):
        cw = CodegenWidget()
        pid = PIDController(Kp=2.0, Ti=1.0, Td=0.1)
        cw.set_pid(pid)
        assert cw._pid is pid

    def test_set_plant_tf(self):
        cw = CodegenWidget()
        G = TransferFunction([1], [1, 1])
        cw.set_plant_tf(G)
        assert cw._plant_tf is G

    def test_make_config_defaults(self):
        cw = CodegenWidget()
        cfg = cw._make_config()
        assert cfg.add_comments is True

    def test_make_config_with_options(self):
        cw = CodegenWidget()
        cw._uart_check.setChecked(True)
        cw._watchdog_check.setChecked(True)
        cfg = cw._make_config()
        assert cfg.include_uart is True
        assert cfg.include_watchdog is True

    def test_make_config_no_comments(self):
        cw = CodegenWidget()
        cw._comments_check.setChecked(False)
        cfg = cw._make_config()
        assert cfg.add_comments is False

    def test_generate_pid_mode(self):
        cw = CodegenWidget()
        cw._mode_combo.setCurrentIndex(0)  # "PID-Regler"
        cw._on_generate()
        code = cw._code_edit.toPlainText()
        assert "class PIDController" in code

    def test_generate_digital_filter_mode_no_plant(self):
        """Without plant, a warning message box should appear."""
        cw = CodegenWidget()
        cw._mode_combo.setCurrentIndex(1)  # "Digitaler Filter"
        cw._plant_tf = None
        with patch("pylabb.gui.widgets.codegen_widget.QMessageBox.warning"):
            cw._on_generate()
        # code should remain empty
        assert cw._code_edit.toPlainText() == ""

    def test_generate_digital_filter_mode_with_plant(self):
        cw = CodegenWidget()
        cw._mode_combo.setCurrentIndex(1)
        cw.set_plant_tf(TransferFunction([1], [1, 1]))
        cw._on_generate()
        assert "class DigitalFilter" in cw._code_edit.toPlainText()

    def test_generate_full_loop_mode(self):
        cw = CodegenWidget()
        cw._mode_combo.setCurrentIndex(2)  # "Vollständige Hauptschleife"
        cw._on_generate()
        assert "while True" in cw._code_edit.toPlainText()

    def test_on_copy(self):
        cw = CodegenWidget()
        cw._code_edit.setPlainText("test_code = 1")
        cw._on_copy()
        from PyQt6.QtWidgets import QApplication
        assert QApplication.clipboard().text() == "test_code = 1"

    def test_code_generated_signal(self):
        received = []
        cw = CodegenWidget()
        cw.code_generated.connect(lambda c: received.append(c))
        cw._mode_combo.setCurrentIndex(0)
        cw._on_generate()
        assert len(received) == 1
        assert "class" in received[0]

    def test_export_project_no_dir(self):
        """If directory dialog is cancelled, nothing happens."""
        cw = CodegenWidget()
        with patch("pylabb.gui.widgets.codegen_widget.QFileDialog.getExistingDirectory",
                   return_value=""):
            cw._on_export_project()  # should not raise

    def test_export_project_with_dir(self, tmp_path):
        cw = CodegenWidget()
        with patch("pylabb.gui.widgets.codegen_widget.QFileDialog.getExistingDirectory",
                   return_value=str(tmp_path)):
            with patch("pylabb.gui.widgets.codegen_widget.QMessageBox.information"):
                cw._on_export_project()
        assert (tmp_path / "pid.py").exists()
        assert (tmp_path / "main.py").exists()


# =========================================================================
# system_editor.py – standalone helper
# =========================================================================

class TestParseCoeffs:
    def test_space_separated(self):
        assert _parse_coeffs("1 2 3") == [1.0, 2.0, 3.0]

    def test_comma_separated(self):
        assert _parse_coeffs("1, 2, 3") == [1.0, 2.0, 3.0]

    def test_mixed(self):
        assert _parse_coeffs("1, 2 3") == [1.0, 2.0, 3.0]

    def test_single(self):
        assert _parse_coeffs("5") == [5.0]


# =========================================================================
# TransferFunctionEditor
# =========================================================================

class TestTransferFunctionEditor:
    def test_construction(self):
        ed = TransferFunctionEditor()
        assert ed is not None

    def test_get_tf_default(self):
        ed = TransferFunctionEditor()
        tf = ed.get_tf()
        assert tf is not None
        # Default: num=1, den=1 1
        assert len(tf.num) == 1
        assert len(tf.den) == 2

    def test_on_apply_valid(self):
        ed = TransferFunctionEditor()
        received = []
        ed.tf_changed.connect(lambda tf: received.append(tf))
        ed._num_edit.setText("2")
        ed._den_edit.setText("1 3 2")
        ed._on_apply()
        assert len(received) == 1
        assert abs(received[0].dc_gain() - 1.0) < 1e-6

    def test_on_apply_invalid(self):
        ed = TransferFunctionEditor()
        ed._num_edit.setText("not_a_number")
        ed._on_apply()
        assert "Fehler" in ed._status_label.text()

    def test_on_reset(self):
        ed = TransferFunctionEditor()
        ed._num_edit.setText("5 3 1")
        ed._on_reset()
        assert ed._num_edit.text() == "1"
        assert ed._den_edit.text() == "1 1"

    def test_set_tf(self):
        ed = TransferFunctionEditor()
        G = TransferFunction([2], [1, 3], name="MyG")
        ed.set_tf(G)
        assert "2" in ed._num_edit.text()
        assert ed._name_edit.text() == "MyG"

    def test_set_tf_discrete(self):
        ed = TransferFunctionEditor()
        G = TransferFunction([1], [1, 1]).discretize(0.01)
        ed.set_tf(G)
        assert ed._dt_check.isChecked()

    def test_get_tf_with_dt(self):
        ed = TransferFunctionEditor()
        ed._dt_check.setChecked(True)
        ed._dt_spin.setValue(0.01)
        tf = ed.get_tf()
        assert tf.dt == pytest.approx(0.01, abs=1e-9)

    def test_get_tf_invalid_returns_none(self):
        ed = TransferFunctionEditor()
        ed._num_edit.setText("bad")
        assert ed.get_tf() is None


# =========================================================================
# PIDEditor
# =========================================================================

class TestPIDEditor:
    def test_construction(self):
        pe = PIDEditor()
        assert pe is not None

    def test_get_pid_default(self):
        pe = PIDEditor()
        pid = pe.get_pid()
        assert pid is not None

    def test_get_discrete_pid(self):
        pe = PIDEditor()
        dpid = pe.get_discrete_pid()
        assert dpid is not None

    def test_on_apply(self):
        pe = PIDEditor()
        received = []
        pe.pid_changed.connect(lambda p: received.append(p))
        pe._kp_spin.setValue(3.0)
        pe._on_apply()
        assert len(received) == 1
        assert received[0].Kp == pytest.approx(3.0)

    def test_on_reset(self):
        pe = PIDEditor()
        pe._kp_spin.setValue(5.0)
        pe._on_reset()
        assert pe._kp_spin.value() == pytest.approx(1.0)

    def test_tune_zn_step(self):
        pe = PIDEditor()
        pe._rule_combo.setCurrentText("ZN-Sprung")
        pe._K_spin.setValue(1.0)
        pe._T_spin.setValue(2.0)
        pe._L_spin.setValue(0.2)
        pe._type_combo.setCurrentText("PID")
        pe._on_tune()
        assert "ZN-Sprung" in pe._status.text()

    def test_tune_zn_oscillation(self):
        pe = PIDEditor()
        pe._rule_combo.setCurrentText("ZN-Schwingung")
        pe._K_spin.setValue(2.0)
        pe._T_spin.setValue(1.0)
        pe._type_combo.setCurrentText("PI")
        pe._on_tune()
        assert "ZN-Schwingung" in pe._status.text()

    def test_tune_cohen_coon(self):
        pe = PIDEditor()
        pe._rule_combo.setCurrentText("Cohen-Coon")
        pe._K_spin.setValue(1.0)
        pe._T_spin.setValue(2.0)
        pe._L_spin.setValue(0.2)
        pe._on_tune()
        assert "Cohen-Coon" in pe._status.text()

    def test_tune_lambda(self):
        pe = PIDEditor()
        pe._rule_combo.setCurrentText("Lambda")
        pe._K_spin.setValue(1.0)
        pe._T_spin.setValue(2.0)
        pe._L_spin.setValue(0.2)
        pe._on_tune()
        assert "Lambda" in pe._status.text()

    def test_tune_invalid_raises_shown(self):
        pe = PIDEditor()
        pe._rule_combo.setCurrentText("ZN-Sprung")
        pe._K_spin.setValue(0.0)    # K=0 → division may fail
        pe._T_spin.setValue(0.0)
        pe._L_spin.setValue(0.0)
        pe._on_tune()  # should show error, not raise


# =========================================================================
# SystemEditorPanel
# =========================================================================

class TestSystemEditorPanel:
    def test_construction(self):
        sep = SystemEditorPanel()
        assert sep is not None

    def test_get_plant_default(self):
        sep = SystemEditorPanel()
        tf = sep.get_plant()
        assert tf is not None

    def test_get_pid(self):
        sep = SystemEditorPanel()
        pid = sep.get_pid()
        assert isinstance(pid, PIDController)

    def test_plant_changed_signal(self):
        sep = SystemEditorPanel()
        received = []
        sep.plant_changed.connect(lambda tf: received.append(tf))
        sep.tf_editor._num_edit.setText("2")
        sep.tf_editor._on_apply()
        assert len(received) >= 1

    def test_controller_changed_signal(self):
        sep = SystemEditorPanel()
        received = []
        sep.controller_changed.connect(lambda p: received.append(p))
        sep.pid_editor._on_apply()
        assert len(received) == 1

    def test_standard_system_button_emits(self):
        """Click first standard-system button → plant_changed emitted."""
        sep = SystemEditorPanel()
        received = []
        sep.plant_changed.connect(lambda tf: received.append(tf))
        # Find the standard-systems tab (index 2) and click its first button
        from PyQt6.QtWidgets import QPushButton
        # Access via the tab widget
        tab_widget = sep.findChild(
            __import__("PyQt6.QtWidgets", fromlist=["QTabWidget"]).QTabWidget
        )
        std_widget = tab_widget.widget(2)
        buttons = std_widget.findChildren(QPushButton)
        if buttons:
            buttons[0].click()
            assert len(received) >= 1


# =========================================================================
# MainWindow
# =========================================================================

class TestMainWindow:
    @pytest.fixture
    def mw(self):
        """Create a fresh MainWindow for each test."""
        with patch("pylabb.gui.main_window.QMessageBox.about"):
            w = MainWindow()
        return w

    def test_construction(self, mw):
        assert mw is not None
        assert mw._plant is not None

    def test_title(self, mw):
        assert "PyLab" in mw.windowTitle()

    def test_default_system_loaded(self, mw):
        assert mw._plant is not None
        assert mw._controller is not None

    def test_on_plant_changed(self, mw):
        G = TransferFunction([2], [1, 2])
        mw._on_plant_changed(G)
        assert mw._plant is G

    def test_on_controller_changed(self, mw):
        pid = PIDController(Kp=3.0, Ti=2.0, Td=0.0)
        mw._on_controller_changed(pid)
        assert mw._controller is pid

    def test_on_run_analysis(self, mw):
        mw._on_run_analysis()
        assert mw._analysis_widget.last_info is not None

    def test_on_run_analysis_no_plant(self, mw):
        mw._plant = None
        with patch.object(mw, "_warn"):
            mw._on_run_analysis()

    def test_on_plot_bode(self, mw):
        mw._on_plot_bode()

    def test_on_plot_bode_no_plant(self, mw):
        mw._plant = None
        with patch.object(mw, "_warn"):
            mw._on_plot_bode()

    def test_on_plot_nyquist(self, mw):
        mw._on_plot_nyquist()

    def test_on_plot_nyquist_no_plant(self, mw):
        mw._plant = None
        with patch.object(mw, "_warn"):
            mw._on_plot_nyquist()

    def test_on_plot_rlocus(self, mw):
        mw._on_plot_rlocus()

    def test_on_plot_rlocus_no_plant(self, mw):
        mw._plant = None
        with patch.object(mw, "_warn"):
            mw._on_plot_rlocus()

    def test_on_plot_step(self, mw):
        mw._on_plot_step()

    def test_on_plot_step_no_plant(self, mw):
        mw._plant = None
        with patch.object(mw, "_warn"):
            mw._on_plot_step()

    def test_on_simulate_closed_loop(self, mw):
        mw._on_simulate_closed_loop()

    def test_on_simulate_no_plant(self, mw):
        mw._plant = None
        with patch.object(mw, "_warn"):
            mw._on_simulate_closed_loop()

    def test_on_simulate_no_controller(self, mw):
        mw._controller = None
        with patch.object(mw, "_warn"):
            mw._on_simulate_closed_loop()

    def test_on_codegen(self, mw):
        mw._on_codegen()
        code = mw._codegen_widget._code_edit.toPlainText()
        assert "class" in code

    def test_on_new_project(self, mw):
        mw._on_new_project()
        assert mw._plant is None
        assert mw._controller is None

    def test_on_about(self, mw):
        with patch("pylabb.gui.main_window.QMessageBox.about"):
            mw._on_about()

    def test_on_export_no_dir(self, mw):
        with patch("pylabb.gui.widgets.codegen_widget.QFileDialog.getExistingDirectory",
                   return_value=""):
            mw._on_export()

    def test_status_method(self, mw):
        mw._status("test_message")
        assert mw._status_label.text() == "test_message"

    def test_check_plant_true(self, mw):
        assert mw._check_plant() is True

    def test_check_plant_false(self, mw):
        mw._plant = None
        with patch.object(mw, "_warn"):
            result = mw._check_plant()
        assert result is False

    def test_warn_method(self, mw):
        with patch("pylabb.gui.main_window.QMessageBox.warning"):
            mw._warn("title", "msg")

    def test_error_method(self, mw):
        with patch("pylabb.gui.main_window.QMessageBox.critical"):
            mw._error("title", "msg")

"""
tests/test_bio_classify_widget.py
====================================
100 % Testabdeckung für pylabb.gui.widgets.bio_classify_widget.

Geprüfte Klassen/Funktionen
-----------------------------
* _parse_coeffs
* _BioBadge         – _set_neutral, update_class (alle BioClass-Werte + Fallback-Key)
* _MatrixView       – set_matrix (0- und 1-Einträge), clear_matrix
* BioClassifyWidget – Initialisierung, _on_classify (Erfolg, ValueError, generische
                      Exception), _on_clear, _display_result (leere/volle
                      subgraph_chain mit/ohne __name__, mit/ohne Erweiterungen,
                      Erweiterung mit/ohne code_hint), set_loop,
                      last_result-Property, classification_done-Signal
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.bio_classify import BioClass, BioClassResult, BioExtension

from pylabb.gui.widgets.bio_classify_widget import (
    BioClassifyWidget,
    _BioBadge,
    _MatrixView,
    _parse_coeffs,
)

_CLASSIFY_PATH = "pylabb.gui.widgets.bio_classify_widget.classify_loop"
_GET_EXT_PATH = "pylabb.gui.widgets.bio_classify_widget.get_extensions"


# ---------------------------------------------------------------------------
# QApplication (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Hilfsfunktion: BioClassResult erstellen
# ---------------------------------------------------------------------------

def _make_result(
    bio_class: BioClass = BioClass.SEESTERN,
    confidence: float = 0.9,
    chain: list | None = None,
) -> BioClassResult:
    adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=int)
    return BioClassResult(
        bio_class=bio_class,
        confidence=confidence,
        subgraph_chain=chain if chain is not None else [],
        adjacency_matrix=adj,
        density=float(adj.sum()) / adj.size,
        has_self_loop=False,
        has_full_last_row=False,
        controller_has_integrator=False,
        controller_has_derivative=False,
    )


# ---------------------------------------------------------------------------
# _parse_coeffs
# ---------------------------------------------------------------------------

class TestParseCoeffs:
    def test_space_separated(self) -> None:
        assert _parse_coeffs("1 2 3") == [1.0, 2.0, 3.0]

    def test_comma_separated(self) -> None:
        assert _parse_coeffs("1, 2, 3") == [1.0, 2.0, 3.0]

    def test_mixed_separators(self) -> None:
        assert _parse_coeffs("1, 2 3") == [1.0, 2.0, 3.0]

    def test_single_value(self) -> None:
        assert _parse_coeffs("5") == [5.0]

    def test_float_values(self) -> None:
        assert _parse_coeffs("1.5 2.0 -3.0") == pytest.approx([1.5, 2.0, -3.0])


# ---------------------------------------------------------------------------
# _BioBadge
# ---------------------------------------------------------------------------

class TestBioBadge:
    @pytest.fixture()
    def badge(self) -> _BioBadge:
        return _BioBadge()

    def test_initial_text_is_dash(self, badge: _BioBadge) -> None:
        assert "–" in badge.text()

    def test_set_neutral_restores_dash(self, badge: _BioBadge) -> None:
        badge.update_class(BioClass.SEESTERN, 0.9)
        badge._set_neutral()
        assert "–" in badge.text()

    def test_initial_stylesheet_has_background(self, badge: _BioBadge) -> None:
        assert "background-color" in badge.styleSheet()

    @pytest.mark.parametrize("bio_class", list(BioClass))
    def test_update_all_bio_classes(
        self, badge: _BioBadge, bio_class: BioClass
    ) -> None:
        badge.update_class(bio_class, 0.8)
        assert "background-color" in badge.styleSheet()
        assert "%" in badge.text()

    def test_update_unknown_key_uses_fallback(self, badge: _BioBadge) -> None:
        fake_class = MagicMock()
        fake_class.name = "NONEXISTENT_CLASS"
        badge.update_class(fake_class, 0.5)
        assert "NONEXISTENT_CLASS" in badge.text()

    def test_update_sets_confidence_percentage(self, badge: _BioBadge) -> None:
        badge.update_class(BioClass.HERZ, 0.75)
        assert "75%" in badge.text()

    def test_stylesheet_contains_color_after_update(self, badge: _BioBadge) -> None:
        badge.update_class(BioClass.TINTENFISCH, 1.0)
        assert "#2c3e50" in badge.styleSheet()


# ---------------------------------------------------------------------------
# _MatrixView
# ---------------------------------------------------------------------------

class TestMatrixView:
    @pytest.fixture()
    def view(self) -> _MatrixView:
        return _MatrixView("Test-Matrix")

    def test_set_matrix_rowcount(self, view: _MatrixView) -> None:
        view.set_matrix(np.array([[0, 1], [1, 0]], dtype=int))
        assert view._table.rowCount() == 2

    def test_set_matrix_columncount(self, view: _MatrixView) -> None:
        view.set_matrix(np.array([[0, 1], [1, 0]], dtype=int))
        assert view._table.columnCount() == 2

    def test_set_matrix_value_zero(self, view: _MatrixView) -> None:
        view.set_matrix(np.array([[0]], dtype=int))
        assert view._table.item(0, 0).text() == "0"

    def test_set_matrix_value_one(self, view: _MatrixView) -> None:
        view.set_matrix(np.array([[1]], dtype=int))
        assert view._table.item(0, 0).text() == "1"

    def test_set_matrix_one_entry_has_green_background(self, view: _MatrixView) -> None:
        from PyQt6.QtGui import QColor
        view.set_matrix(np.array([[1]], dtype=int))
        bg = view._table.item(0, 0).background().color()
        assert bg == QColor("#d5e8d4")

    def test_set_matrix_zero_entry_background_not_green(
        self, view: _MatrixView
    ) -> None:
        from PyQt6.QtGui import QColor
        view.set_matrix(np.array([[0]], dtype=int))
        bg = view._table.item(0, 0).background().color()
        assert bg != QColor("#d5e8d4")

    def test_set_matrix_header_labels(self, view: _MatrixView) -> None:
        view.set_matrix(np.array([[0, 1], [1, 0]], dtype=int))
        assert view._table.horizontalHeaderItem(0).text() == "x0"
        assert view._table.horizontalHeaderItem(1).text() == "x1"

    def test_set_matrix_vertical_labels(self, view: _MatrixView) -> None:
        view.set_matrix(np.array([[0, 1], [1, 0]], dtype=int))
        assert view._table.verticalHeaderItem(0).text() == "x0"

    def test_clear_matrix_resets_counts(self, view: _MatrixView) -> None:
        view.set_matrix(np.array([[1, 0], [0, 1]], dtype=int))
        view.clear_matrix()
        assert view._table.rowCount() == 0
        assert view._table.columnCount() == 0

    def test_set_4x4_matrix(self, view: _MatrixView) -> None:
        mat = np.eye(4, dtype=int)
        view.set_matrix(mat)
        assert view._table.rowCount() == 4


# ---------------------------------------------------------------------------
# BioClassifyWidget
# ---------------------------------------------------------------------------

class TestBioClassifyWidget:
    @pytest.fixture()
    def widget(self) -> BioClassifyWidget:
        return BioClassifyWidget()

    # -- Initialisierung -------------------------------------------------

    def test_initial_last_result_is_none(self, widget: BioClassifyWidget) -> None:
        assert widget.last_result is None

    def test_widget_has_classify_button(self, widget: BioClassifyWidget) -> None:
        assert widget._classify_btn is not None

    def test_widget_has_clear_button(self, widget: BioClassifyWidget) -> None:
        assert widget._clear_btn is not None

    def test_widget_has_badge(self, widget: BioClassifyWidget) -> None:
        assert isinstance(widget._badge, _BioBadge)

    def test_widget_has_all_checkboxes(self, widget: BioClassifyWidget) -> None:
        for attr in (
            "_cb_feedforward",
            "_cb_cascade",
            "_cb_fsf",
            "_cb_mimo",
            "_cb_hysteresis",
            "_cb_disturbance",
        ):
            assert getattr(widget, attr) is not None

    def test_widget_has_spinboxes(self, widget: BioClassifyWidget) -> None:
        assert widget._sb_hysteresis_band is not None
        assert widget._sb_threshold is not None

    def test_widget_has_result_text(self, widget: BioClassifyWidget) -> None:
        assert widget._result_text is not None

    # -- set_loop --------------------------------------------------------

    def test_set_loop_updates_plant_numerator(self, widget: BioClassifyWidget) -> None:
        G = TransferFunction([3.0, 1.0], [1.0, 2.0, 1.0])
        C = TransferFunction([2.0], [1.0])
        widget.set_loop(G, C)
        assert "3.0" in widget._g_num.text()

    def test_set_loop_updates_plant_denominator(self, widget: BioClassifyWidget) -> None:
        G = TransferFunction([1.0], [1.0, 5.0])
        C = TransferFunction([1.0], [1.0])
        widget.set_loop(G, C)
        assert "5.0" in widget._g_den.text()

    def test_set_loop_updates_controller_numerator(
        self, widget: BioClassifyWidget
    ) -> None:
        G = TransferFunction([1.0], [1.0, 1.0])
        C = TransferFunction([4.0, 1.0], [1.0])
        widget.set_loop(G, C)
        assert "4.0" in widget._c_num.text()

    def test_set_loop_updates_controller_denominator(
        self, widget: BioClassifyWidget
    ) -> None:
        G = TransferFunction([1.0], [1.0, 1.0])
        C = TransferFunction([1.0], [1.0, 2.0])
        widget.set_loop(G, C)
        assert "2.0" in widget._c_den.text()

    # -- _on_classify: Erfolgsfall --------------------------------------

    def test_classify_success_sets_last_result(self, widget: BioClassifyWidget) -> None:
        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result):
            widget._on_classify()
        assert widget.last_result is result

    def test_classify_success_emits_signal(self, widget: BioClassifyWidget) -> None:
        received: list[BioClassResult] = []
        widget.classification_done.connect(received.append)
        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result):
            widget._on_classify()
        assert len(received) == 1
        assert isinstance(received[0], BioClassResult)

    def test_classify_passes_flags_to_api(self, widget: BioClassifyWidget) -> None:
        widget._cb_feedforward.setChecked(True)
        widget._cb_cascade.setChecked(True)
        widget._cb_fsf.setChecked(True)
        widget._cb_mimo.setChecked(True)
        widget._cb_hysteresis.setChecked(True)
        widget._sb_hysteresis_band.setValue(0.2)
        widget._cb_disturbance.setChecked(True)
        widget._sb_threshold.setValue(0.6)

        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result) as mock_classify:
            widget._on_classify()

        _, kwargs = mock_classify.call_args
        assert kwargs["has_feedforward"] is True
        assert kwargs["has_cascade"] is True
        assert kwargs["is_full_state_feedback"] is True
        assert kwargs["is_mimo"] is True
        assert kwargs["has_hysteresis"] is True
        assert kwargs["hysteresis_band"] == pytest.approx(0.2, abs=1e-3)
        assert kwargs["has_disturbance_channel"] is True
        assert kwargs["threshold"] == pytest.approx(0.6, abs=1e-3)

    def test_classify_passes_unchecked_flags(self, widget: BioClassifyWidget) -> None:
        for cb in (
            widget._cb_feedforward,
            widget._cb_cascade,
            widget._cb_fsf,
            widget._cb_mimo,
            widget._cb_hysteresis,
            widget._cb_disturbance,
        ):
            cb.setChecked(False)
        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result) as mock_classify:
            widget._on_classify()
        _, kwargs = mock_classify.call_args
        assert kwargs["has_feedforward"] is False
        assert kwargs["has_cascade"] is False

    def test_classify_plant_and_controller_passed(self, widget: BioClassifyWidget) -> None:
        widget._g_num.setText("2")
        widget._g_den.setText("1 3")
        widget._c_num.setText("5")
        widget._c_den.setText("1")
        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result) as mock_classify:
            widget._on_classify()
        args, _ = mock_classify.call_args
        plant, ctrl = args[0], args[1]
        assert np.allclose(plant.num.real, [2.0], atol=1e-9)
        assert np.allclose(ctrl.num.real, [5.0], atol=1e-9)

    # -- _on_classify: Fehlerbehandlung ---------------------------------

    def test_classify_invalid_g_num_shows_warning(
        self, widget: BioClassifyWidget
    ) -> None:
        widget._g_num.setText("not_a_float")
        with patch(
            "pylabb.gui.widgets.bio_classify_widget.QMessageBox.warning"
        ) as mock_warn:
            widget._on_classify()
        mock_warn.assert_called_once()
        assert widget.last_result is None

    def test_classify_exception_shows_critical(self, widget: BioClassifyWidget) -> None:
        widget._g_num.setText("1")
        with patch(_CLASSIFY_PATH, side_effect=RuntimeError("boom")):
            with patch(
                "pylabb.gui.widgets.bio_classify_widget.QMessageBox.critical"
            ) as mock_crit:
                widget._on_classify()
        mock_crit.assert_called_once()

    def test_classify_exception_does_not_set_last_result(
        self, widget: BioClassifyWidget
    ) -> None:
        widget._g_num.setText("1")
        widget.last_result  # reset via clear
        widget._on_clear()
        with patch(_CLASSIFY_PATH, side_effect=ValueError("bad")):
            with patch("pylabb.gui.widgets.bio_classify_widget.QMessageBox.critical"):
                widget._on_classify()
        assert widget.last_result is None

    # -- _on_clear -------------------------------------------------------

    def test_clear_resets_last_result(self, widget: BioClassifyWidget) -> None:
        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result):
            widget._on_classify()
        assert widget.last_result is not None
        widget._on_clear()
        assert widget.last_result is None

    def test_clear_empties_result_text(self, widget: BioClassifyWidget) -> None:
        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result):
            widget._on_classify()
        widget._on_clear()
        assert widget._result_text.toPlainText() == ""

    def test_clear_resets_chain_label(self, widget: BioClassifyWidget) -> None:
        node = MagicMock()
        node.__name__ = "SomeNode"
        node.label = "SomeNode"
        result = _make_result(chain=[node])
        with patch(_CLASSIFY_PATH, return_value=result):
            widget._on_classify()
        widget._on_clear()
        assert widget._chain_label.text() == "–"

    def test_clear_empties_matrix(self, widget: BioClassifyWidget) -> None:
        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result):
            widget._on_classify()
        widget._on_clear()
        assert widget._mat_view._table.rowCount() == 0

    def test_clear_resets_badge(self, widget: BioClassifyWidget) -> None:
        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result):
            widget._on_classify()
        widget._on_clear()
        assert "–" in widget._badge.text()

    # -- _display_result -------------------------------------------------

    def test_display_result_badge_updated(self, widget: BioClassifyWidget) -> None:
        result = _make_result(BioClass.KLEINHIRN, 0.85)
        with patch(_GET_EXT_PATH, return_value=[]):
            widget._display_result(result)
        assert "%" in widget._badge.text()

    def test_display_result_matrix_set(self, widget: BioClassifyWidget) -> None:
        result = _make_result()
        with patch(_GET_EXT_PATH, return_value=[]):
            widget._display_result(result)
        assert widget._mat_view._table.rowCount() == 3

    def test_display_result_empty_chain_shows_dash(
        self, widget: BioClassifyWidget
    ) -> None:
        result = _make_result(chain=[])
        with patch(_GET_EXT_PATH, return_value=[]):
            widget._display_result(result)
        assert widget._chain_label.text() == "–"

    def test_display_result_chain_with_name_attr(
        self, widget: BioClassifyWidget
    ) -> None:
        node = MagicMock()
        node.__name__ = "NodeAlpha"
        node.label = "NodeAlpha"
        result = _make_result(chain=[node])
        with patch(_GET_EXT_PATH, return_value=[]):
            widget._display_result(result)
        assert "NodeAlpha" in widget._chain_label.text()

    def test_display_result_chain_without_name_attr(
        self, widget: BioClassifyWidget
    ) -> None:
        class _PlainNode:
            label = "plain_node_str"

            def __str__(self) -> str:
                return "plain_node_str"

        result = _make_result(chain=[_PlainNode()])
        with patch(_GET_EXT_PATH, return_value=[]):
            widget._display_result(result)
        assert "plain_node_str" in widget._chain_label.text()

    def test_display_result_shows_extension_name(
        self, widget: BioClassifyWidget
    ) -> None:
        ext = BioExtension(name="Konsens-P", description="Schwarm-Regler", code_hint="")
        result = _make_result()
        with patch(_GET_EXT_PATH, return_value=[ext]):
            widget._display_result(result)
        assert "Konsens-P" in widget._result_text.toHtml()

    def test_display_result_extension_with_code_hint_shown(
        self, widget: BioClassifyWidget
    ) -> None:
        ext = BioExtension(name="Ext", description="Desc", code_hint="my_func(x)")
        result = _make_result()
        with patch(_GET_EXT_PATH, return_value=[ext]):
            widget._display_result(result)
        assert "my_func(x)" in widget._result_text.toHtml()

    def test_display_result_extension_empty_code_hint_not_shown(
        self, widget: BioClassifyWidget
    ) -> None:
        ext = BioExtension(name="Ext2", description="Desc2", code_hint="")
        result = _make_result()
        with patch(_GET_EXT_PATH, return_value=[ext]):
            widget._display_result(result)
        assert "Hinweis:" not in widget._result_text.toPlainText()

    def test_display_result_no_extensions_no_section_header(
        self, widget: BioClassifyWidget
    ) -> None:
        result = _make_result()
        with patch(_GET_EXT_PATH, return_value=[]):
            widget._display_result(result)
        assert "Erweiterungsvorschläge" not in widget._result_text.toPlainText()

    def test_display_result_has_summary_text(self, widget: BioClassifyWidget) -> None:
        result = _make_result()
        with patch(_GET_EXT_PATH, return_value=[]):
            widget._display_result(result)
        assert "Zusammenfassung" in widget._result_text.toHtml()

    def test_display_result_shows_density(self, widget: BioClassifyWidget) -> None:
        result = _make_result()
        with patch(_GET_EXT_PATH, return_value=[]):
            widget._display_result(result)
        assert "Dichte" in widget._result_text.toHtml()

    def test_display_result_via_classify_button(
        self, widget: BioClassifyWidget
    ) -> None:
        """Integration: classify button → _display_result called."""
        widget._g_num.setText("1")
        widget._g_den.setText("1 1.4 1")
        widget._c_num.setText("1 1")
        widget._c_den.setText("1")
        result = _make_result(BioClass.HERZ, 0.7)
        with patch(_CLASSIFY_PATH, return_value=result):
            widget._classify_btn.click()
        assert "%" in widget._badge.text()

    # -- last_result property --------------------------------------------

    def test_last_result_property_none_initially(
        self, widget: BioClassifyWidget
    ) -> None:
        assert widget.last_result is None

    def test_last_result_property_set_after_classify(
        self, widget: BioClassifyWidget
    ) -> None:
        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result):
            widget._on_classify()
        assert widget.last_result is result

    def test_last_result_property_cleared_after_clear(
        self, widget: BioClassifyWidget
    ) -> None:
        result = _make_result()
        with patch(_CLASSIFY_PATH, return_value=result):
            widget._on_classify()
        widget._on_clear()
        assert widget.last_result is None


# ---------------------------------------------------------------------------
# Neue MainWindow-Funktionalität (bio_classify-Integration)
# ---------------------------------------------------------------------------

class TestMainWindowBioClassify:
    """Tests für die in main_window.py hinzugefügten Bio-Klassifikations-Slots."""

    @pytest.fixture()
    def mw(self):
        from unittest.mock import patch as _patch
        from pylabb.gui.main_window import MainWindow
        with _patch("pylabb.gui.main_window.QMessageBox.about"):
            w = MainWindow()
        return w

    def test_bio_classify_widget_created(self, mw) -> None:
        assert isinstance(mw._bio_classify_widget, BioClassifyWidget)

    def test_bio_dock_created(self, mw) -> None:
        from PyQt6.QtWidgets import QDockWidget
        assert isinstance(mw._dock_bio, QDockWidget)

    def test_act_bio_classify_action_exists(self, mw) -> None:
        assert mw._act_bio_classify is not None

    def test_on_bio_classify_shows_dock(self, mw) -> None:
        mw.show()
        mw._on_bio_classify()
        assert mw._dock_bio.isVisible()

    def test_on_load_current_loop_shows_dock(self, mw) -> None:
        mw.show()
        mw._on_load_current_loop()
        assert mw._dock_bio.isVisible()

    def test_on_load_current_loop_no_plant_does_not_crash(self, mw) -> None:
        mw._plant = None
        mw._on_load_current_loop()  # _sync_bio_classify_loop returns early

    def test_on_load_current_loop_updates_status(self, mw) -> None:
        mw._on_load_current_loop()
        assert "Bio-Klassifikation" in mw._status_label.text()

    def test_sync_bio_classify_loop_with_controller(self, mw) -> None:
        from pylabb.core.transfer_function import TransferFunction
        from pylabb.control.pid import PIDController
        mw._plant = TransferFunction([1.0], [1.0, 1.0])
        mw._controller = PIDController(Kp=2.0, Ti=1.0, Td=0.1)
        mw._sync_bio_classify_loop()
        # Strecken-Zähler muss im Widget erscheinen
        assert "1.0" in mw._bio_classify_widget._g_num.text()

    def test_sync_bio_classify_loop_no_controller(self, mw) -> None:
        from pylabb.core.transfer_function import TransferFunction
        mw._plant = TransferFunction([1.0], [1.0, 2.0])
        mw._controller = None
        mw._sync_bio_classify_loop()
        # Default-Regler [1.0]/[1.0] verwendet
        assert "1.0" in mw._bio_classify_widget._c_num.text()

    def test_sync_bio_classify_loop_no_plant_returns_early(self, mw) -> None:
        mw._plant = None
        mw._sync_bio_classify_loop()  # darf nicht crashen

    def test_classification_done_signal_updates_status(self, mw) -> None:
        result = _make_result(BioClass.PUPILLE, 0.95)
        mw._bio_classify_widget.classification_done.emit(result)
        text = mw._status_label.text()
        assert "Bio-Klasse" in text
        assert "95%" in text

    def test_on_load_current_as_a_also_syncs_bio(self, mw) -> None:
        """_on_load_current_as_A soll jetzt auch Bio-Widget synchronisieren."""
        from pylabb.core.transfer_function import TransferFunction
        from pylabb.control.pid import PIDController
        mw._plant = TransferFunction([2.0], [1.0, 3.0])
        mw._controller = PIDController(Kp=1.0)
        mw._on_load_current_as_A()
        # Bio-Klassifikations-Widget muss Strecke kennen
        assert "2.0" in mw._bio_classify_widget._g_num.text()

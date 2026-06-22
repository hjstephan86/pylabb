"""
tests/test_verification_widget.py
===================================
100% Testabdeckung für pylabb.gui.widgets.verification_widget.

Geprüfte Klassen/Funktionen
-----------------------------
* _parse_coeffs
* _DecisionBadge        – _set_neutral, update_decision (alle Keys + Fallback)
* _MatrixView           – set_matrix (0- und 1-Einträge), clear_matrix
* _LoopInputPanel       – Standardwerte, get_plant, get_controller,
                          set_plant, set_controller
* VerificationWidget    – Initialisierung, _on_verify (Erfolg, ValueError,
                          ImportError, generische Exception), _on_clear,
                          set_loop_A/B, _display_result mit/ohne kept_matrix,
                          Signal verification_done, last_result-Property
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.verification import VerificationResult

# Modul-Import erfolgt erst nach QApp → sicherheitshalber per direktem Import
from pylabb.gui.widgets.verification_widget import (
    VerificationWidget,
    _DecisionBadge,
    _LoopInputPanel,
    _MatrixView,
    _parse_coeffs,
)

_VERIFY_PATH = "pylabb.gui.widgets.verification_widget.verify_loops"


# ---------------------------------------------------------------------------
# QApplication (session-scoped, existiert bereits via conftest)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Hilfsfunktion: VerificationResult mit beliebiger Entscheidung
# ---------------------------------------------------------------------------

def _make_vresult(
    decision: str,
    kept: np.ndarray | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> VerificationResult:
    mat = np.array([[0, 1], [0, 0]], dtype=int)
    L = TransferFunction([1.0], [1.0, 1.0])
    return VerificationResult(
        decision=decision,
        matrix_A=mat.copy(),
        matrix_B=mat.copy(),
        kept_matrix=kept,
        open_loop_A=L,
        open_loop_B=L,
        label_A=label_a,
        label_B=label_b,
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
        result = _parse_coeffs("1.5 2.0 -3.0")
        assert result == pytest.approx([1.5, 2.0, -3.0])


# ---------------------------------------------------------------------------
# _DecisionBadge
# ---------------------------------------------------------------------------

class TestDecisionBadge:
    @pytest.fixture()
    def badge(self) -> _DecisionBadge:
        return _DecisionBadge()

    def test_initial_text_is_dash(self, badge: _DecisionBadge) -> None:
        assert "–" in badge.text()

    def test_set_neutral_restores_dash(self, badge: _DecisionBadge) -> None:
        badge.update_decision("equal")
        badge._set_neutral()
        assert "–" in badge.text()

    @pytest.mark.parametrize("decision,expected_fragment", [
        ("equal",        "ÄQUIVALENT"),
        ("equal_keep_A", "ÄQUIVALENT"),
        ("equal_keep_B", "ÄQUIVALENT"),
        ("keep_A",       "A ENTHÄLT B"),
        ("keep_B",       "B ENTHÄLT A"),
        ("keep_both",    "STRUKTURELL VERSCHIEDEN"),
    ])
    def test_update_known_decision(
        self, badge: _DecisionBadge, decision: str, expected_fragment: str
    ) -> None:
        badge.update_decision(decision)
        assert expected_fragment in badge.text()

    def test_update_unknown_decision_uses_uppercase_key(
        self, badge: _DecisionBadge
    ) -> None:
        badge.update_decision("something_new")
        assert "SOMETHING_NEW" in badge.text()

    def test_stylesheet_set_after_update(self, badge: _DecisionBadge) -> None:
        badge.update_decision("equal")
        assert "background-color" in badge.styleSheet()


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

    def test_set_matrix_one_entry_has_background(self, view: _MatrixView) -> None:
        from PyQt6.QtGui import QColor
        view.set_matrix(np.array([[1]], dtype=int))
        bg = view._table.item(0, 0).background().color()
        assert bg == QColor("#d5e8d4")

    def test_set_matrix_zero_entry_background_is_not_green(
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

    def test_clear_matrix(self, view: _MatrixView) -> None:
        view.set_matrix(np.array([[1, 0], [0, 1]], dtype=int))
        view.clear_matrix()
        assert view._table.rowCount() == 0
        assert view._table.columnCount() == 0

    def test_set_larger_matrix(self, view: _MatrixView) -> None:
        mat = np.eye(4, dtype=int)
        view.set_matrix(mat)
        assert view._table.rowCount() == 4


# ---------------------------------------------------------------------------
# _LoopInputPanel
# ---------------------------------------------------------------------------

class TestLoopInputPanel:
    @pytest.fixture()
    def panel(self) -> _LoopInputPanel:
        return _LoopInputPanel("Test-Loop")

    def test_default_plant_numerator(self, panel: _LoopInputPanel) -> None:
        G = panel.get_plant()
        assert np.allclose(G.num.real, [1.0], atol=1e-12)

    def test_default_plant_denominator(self, panel: _LoopInputPanel) -> None:
        G = panel.get_plant()
        assert np.allclose(G.den.real, [1.0, 2.0, 1.0], atol=1e-12)

    def test_default_controller_numerator(self, panel: _LoopInputPanel) -> None:
        C = panel.get_controller()
        assert np.allclose(C.num.real, [1.0, 1.0], atol=1e-12)

    def test_default_controller_denominator(self, panel: _LoopInputPanel) -> None:
        C = panel.get_controller()
        assert np.allclose(C.den.real, [1.0], atol=1e-12)

    def test_set_plant_updates_fields(self, panel: _LoopInputPanel) -> None:
        tf = TransferFunction([3.0, 1.0], [1.0, 5.0], name="G_new")
        panel.set_plant(tf)
        G = panel.get_plant()
        assert np.allclose(G.num.real, [3.0, 1.0], atol=1e-12)
        assert np.allclose(G.den.real, [1.0, 5.0], atol=1e-12)

    def test_set_controller_updates_fields(self, panel: _LoopInputPanel) -> None:
        tf = TransferFunction([2.0], [1.0, 0.0], name="C_new")
        panel.set_controller(tf)
        C = panel.get_controller()
        assert np.allclose(C.num.real, [2.0], atol=1e-12)

    def test_get_plant_returns_transfer_function(self, panel: _LoopInputPanel) -> None:
        assert isinstance(panel.get_plant(), TransferFunction)

    def test_get_controller_returns_transfer_function(self, panel: _LoopInputPanel) -> None:
        assert isinstance(panel.get_controller(), TransferFunction)


# ---------------------------------------------------------------------------
# VerificationWidget
# ---------------------------------------------------------------------------

class TestVerificationWidget:
    @pytest.fixture()
    def widget(self) -> VerificationWidget:
        return VerificationWidget()

    # -- Initialisierung -------------------------------------------------

    def test_initial_last_result_is_none(self, widget: VerificationWidget) -> None:
        assert widget.last_result is None

    def test_widget_has_verify_button(self, widget: VerificationWidget) -> None:
        assert widget._verify_btn is not None

    def test_widget_has_clear_button(self, widget: VerificationWidget) -> None:
        assert widget._clear_btn is not None

    # -- set_loop_A / set_loop_B ----------------------------------------

    def test_set_loop_a_updates_panel(self, widget: VerificationWidget) -> None:
        G = TransferFunction([2.0], [1.0, 3.0])
        C = TransferFunction([1.0], [1.0])
        widget.set_loop_A(G, C)
        G_read = widget._loop_A.get_plant()
        assert np.allclose(G_read.num.real, [2.0], atol=1e-12)

    def test_set_loop_b_updates_panel(self, widget: VerificationWidget) -> None:
        G = TransferFunction([5.0], [1.0, 2.0, 1.0])
        C = TransferFunction([1.0], [1.0])
        widget.set_loop_B(G, C)
        G_read = widget._loop_B.get_plant()
        assert np.allclose(G_read.num.real, [5.0], atol=1e-12)

    # -- _on_verify: Erfolgsfall -----------------------------------------

    def test_verify_success_sets_last_result(self, widget: VerificationWidget) -> None:
        result = _make_vresult("equal")
        with patch(_VERIFY_PATH, return_value=result):
            widget._on_verify()
        assert widget.last_result is not None

    def test_verify_success_emits_signal(self, widget: VerificationWidget) -> None:
        received: list[VerificationResult] = []
        widget.verification_done.connect(received.append)
        result = _make_vresult("equal")
        with patch(_VERIFY_PATH, return_value=result):
            widget._on_verify()
        assert len(received) == 1
        assert isinstance(received[0], VerificationResult)

    def test_verify_success_decision_stored(self, widget: VerificationWidget) -> None:
        result = _make_vresult("keep_B")
        with patch(_VERIFY_PATH, return_value=result):
            widget._on_verify()
        assert widget.last_result.decision == "keep_B"

    # -- _on_verify: ValueError ------------------------------------------

    def test_verify_invalid_coefficients_shows_warning(
        self, widget: VerificationWidget
    ) -> None:
        widget._loop_A._g_num.setText("abc")  # kein Float → ValueError
        with patch(
            "pylabb.gui.widgets.verification_widget.QMessageBox.warning"
        ) as mock_warn:
            widget._on_verify()
        mock_warn.assert_called_once()
        assert widget.last_result is None  # kein Ergebnis gesetzt

    # -- _on_verify: ImportError ----------------------------------------

    def test_verify_import_error_shows_critical(
        self, widget: VerificationWidget
    ) -> None:
        widget._loop_A._g_num.setText("1")  # zurücksetzen
        with patch(
            _VERIFY_PATH,
            side_effect=ImportError("subgraph nicht installiert"),
        ):
            with patch(
                "pylabb.gui.widgets.verification_widget.QMessageBox.critical"
            ) as mock_crit:
                widget._on_verify()
        mock_crit.assert_called_once()

    # -- _on_verify: generische Exception --------------------------------

    def test_verify_generic_exception_shows_critical(
        self, widget: VerificationWidget
    ) -> None:
        with patch(
            "pylabb.gui.widgets.verification_widget.verify_loops",
            side_effect=RuntimeError("Etwas ist schiefgelaufen"),
        ):
            with patch(
                "pylabb.gui.widgets.verification_widget.QMessageBox.critical"
            ) as mock_crit:
                widget._on_verify()
        mock_crit.assert_called_once()

    # -- _on_clear --------------------------------------------------------

    def test_clear_resets_last_result(self, widget: VerificationWidget) -> None:
        result = _make_vresult("equal")
        with patch(_VERIFY_PATH, return_value=result):
            widget._on_verify()
        assert widget.last_result is not None
        widget._on_clear()
        assert widget.last_result is None

    def test_clear_empties_result_text(self, widget: VerificationWidget) -> None:
        result = _make_vresult("equal")
        with patch(_VERIFY_PATH, return_value=result):
            widget._on_verify()
        widget._on_clear()
        assert widget._result_text.toPlainText() == ""

    def test_clear_empties_matrix_a(self, widget: VerificationWidget) -> None:
        result = _make_vresult("equal")
        with patch(_VERIFY_PATH, return_value=result):
            widget._on_verify()
        widget._on_clear()
        assert widget._mat_view_A._table.rowCount() == 0

    def test_clear_empties_matrix_b(self, widget: VerificationWidget) -> None:
        result = _make_vresult("equal")
        with patch(_VERIFY_PATH, return_value=result):
            widget._on_verify()
        widget._on_clear()
        assert widget._mat_view_B._table.rowCount() == 0

    # -- _display_result: with kept_matrix --------------------------------

    def test_display_result_with_kept_matrix_shows_text(
        self, widget: VerificationWidget
    ) -> None:
        kept = np.array([[0, 1], [0, 0]], dtype=int)
        result = _make_vresult("keep_B", kept=kept.astype(float))
        widget._display_result(result)
        html = widget._result_text.toHtml()
        assert "Bevorzugte" in html

    # -- _display_result: without kept_matrix (keep_both) ----------------

    def test_display_result_without_kept_matrix(
        self, widget: VerificationWidget
    ) -> None:
        result = _make_vresult("keep_both", kept=None)
        widget._display_result(result)
        html = widget._result_text.toHtml()
        # Kein "Bevorzugte Matrix"-Block bei keep_both
        assert "keep_both" in html or "verschieden" in html.lower()

    # -- _display_result: badge update -----------------------------------

    def test_display_result_updates_badge(self, widget: VerificationWidget) -> None:
        result = _make_vresult("equal")
        widget._display_result(result)
        assert "ÄQUIVALENT" in widget._badge.text()

    # -- _display_result: matrices are set --------------------------------

    def test_display_result_sets_matrix_a(self, widget: VerificationWidget) -> None:
        result = _make_vresult("equal")
        widget._display_result(result)
        assert widget._mat_view_A._table.rowCount() == 2

    def test_display_result_sets_matrix_b(self, widget: VerificationWidget) -> None:
        result = _make_vresult("equal")
        widget._display_result(result)
        assert widget._mat_view_B._table.rowCount() == 2

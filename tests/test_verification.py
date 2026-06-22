"""
tests/test_verification.py
==========================
100% Testabdeckung für pylabb.control.verification.

Geprüfte Einheiten
------------------
* _open_loop
* _to_adjacency_matrix  (inkl. Ordnung-0-Fallback und ndim<2-Fallback)
* loop_to_graph
* VerificationResult – alle Properties und summary-Zweige
* verify_loops        – alle Entscheidungspfade, Padding, adj_list-Variante,
                        ImportError bei fehlendem Paket
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.verification import (
    VerificationResult,
    _open_loop,
    _to_adjacency_matrix,
    loop_to_graph,
    verify_loops,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pt1() -> TransferFunction:
    """PT1-Strecke: G(s) = 1/(s+1)."""
    return TransferFunction([1.0], [1.0, 1.0], name="PT1")


@pytest.fixture()
def pt2() -> TransferFunction:
    """PT2-Strecke: G(s) = 1/(s²+1.4s+1)."""
    return TransferFunction([1.0], [1.0, 1.4, 1.0], name="PT2")


@pytest.fixture()
def p_ctrl() -> TransferFunction:
    """P-Regler: C(s) = 2."""
    return TransferFunction([2.0], [1.0], name="C_P")


@pytest.fixture()
def pi_ctrl() -> TransferFunction:
    """PI-Regler: C(s) = (s+1)/s."""
    return TransferFunction([1.0, 1.0], [1.0, 0.0], name="C_PI")


def _make_subgraph_mock(decision: str, kept: np.ndarray | None) -> types.ModuleType:
    """Erzeugt ein gefälschtes ``subgraph``-Modul mit vorgegebener Entscheidung."""
    mock_mod = types.ModuleType("subgraph")
    algo_instance = MagicMock()
    algo_instance.compare_graphs.return_value = (decision, kept)
    algo_instance.compare_graphs_with_adj_list.return_value = (decision, kept)
    SubgraphCls = MagicMock(return_value=algo_instance)
    mock_mod.Subgraph = SubgraphCls
    return mock_mod


# ---------------------------------------------------------------------------
# _open_loop
# ---------------------------------------------------------------------------

class TestOpenLoop:
    def test_multiplies_numerators(self, pt1: TransferFunction, p_ctrl: TransferFunction) -> None:
        L = _open_loop(pt1, p_ctrl)
        # C·G = 2 / (s+1)  → num = [2], den = [1, 1]
        assert np.allclose(L.num.real, [2.0], atol=1e-12)
        assert np.allclose(L.den.real, [1.0, 1.0], atol=1e-12)

    def test_multiplies_higher_order(self, pt2: TransferFunction, pi_ctrl: TransferFunction) -> None:
        L = _open_loop(pt2, pi_ctrl)
        # C(s)·G(s) = (s+1) / (s³ + 1.4s² + s)
        # num = [1, 1], den = [1, 1.4, 1, 0]  (nach polymul)
        assert L.num.shape[0] == 2
        assert L.den.shape[0] == 4

    def test_name_is_L(self, pt1: TransferFunction, p_ctrl: TransferFunction) -> None:
        L = _open_loop(pt1, p_ctrl)
        assert L.name == "L"


# ---------------------------------------------------------------------------
# _to_adjacency_matrix
# ---------------------------------------------------------------------------

class TestToAdjacencyMatrix:
    def test_first_order_system(self, pt1: TransferFunction) -> None:
        adj = _to_adjacency_matrix(pt1)
        assert adj.dtype == int or np.issubdtype(adj.dtype, np.integer)
        assert adj.shape == (1, 1)

    def test_second_order_system(self, pt2: TransferFunction) -> None:
        adj = _to_adjacency_matrix(pt2)
        assert adj.shape == (2, 2)
        # Alle Einträge sind 0 oder 1
        assert set(adj.flatten().tolist()).issubset({0, 1})

    def test_order_zero_fallback(self) -> None:
        """Reines Verstärkungsglied (0. Ordnung): scipy liefert A=[[0]] → adj=[[0]]."""
        tf_gain = TransferFunction([1.0], [1.0], name="Gain")
        adj = _to_adjacency_matrix(tf_gain)
        assert adj.shape == (1, 1)
        # A = [[0.0]] → kein Zustandsübergang → Adjazenz-Eintrag ist 0
        assert adj[0, 0] == 0

    def test_ndim_lt_2_fallback(self, pt1: TransferFunction) -> None:
        """Fallback wenn A.ndim < 2 (wird durch Mock erzeugt)."""
        A_zero_dim = np.array(0.5)  # ndim=0 < 2, size=1 != 0
        dummy = np.array([[0.0]])
        with patch("pylabb.control.verification.sp_sig.tf2ss",
                   return_value=(A_zero_dim, dummy, dummy, dummy)):
            adj = _to_adjacency_matrix(pt1)
        assert adj.shape == (1, 1)
        assert adj[0, 0] == 1

    def test_threshold_filters_small_values(self) -> None:
        """Kleine Einträge unterhalb des Schwellwerts werden auf 0 gesetzt."""
        tf = TransferFunction([1.0], [1.0, 1.0, 1.0], name="T")
        adj_strict = _to_adjacency_matrix(tf, threshold=1e3)  # sehr hoch → alles 0
        assert np.all(adj_strict == 0)

    def test_binary_values_only(self, pt2: TransferFunction) -> None:
        adj = _to_adjacency_matrix(pt2)
        assert np.all((adj == 0) | (adj == 1))


# ---------------------------------------------------------------------------
# loop_to_graph
# ---------------------------------------------------------------------------

class TestLoopToGraph:
    def test_returns_matrix_and_tf(self, pt1: TransferFunction, p_ctrl: TransferFunction) -> None:
        matrix, L = loop_to_graph(pt1, p_ctrl)
        assert isinstance(matrix, np.ndarray)
        assert isinstance(L, TransferFunction)

    def test_matrix_is_square(self, pt2: TransferFunction, p_ctrl: TransferFunction) -> None:
        matrix, _ = loop_to_graph(pt2, p_ctrl)
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]

    def test_open_loop_is_product(self, pt2: TransferFunction, pi_ctrl: TransferFunction) -> None:
        _, L = loop_to_graph(pt2, pi_ctrl)
        # Ordnung = Ordnung(G) + Ordnung(C) = 2 + 1 = 3
        # (Zähler von C ist Grad 1, also offener Kreis Nenner Grad 3)
        assert len(L.den) - 1 >= 2

    def test_custom_threshold_passed(self, pt1: TransferFunction, p_ctrl: TransferFunction) -> None:
        matrix_default, _ = loop_to_graph(pt1, p_ctrl)
        matrix_strict, _ = loop_to_graph(pt1, p_ctrl, threshold=1e9)
        # Mit sehr hohem Schwellwert sollte die Matrix vollständig null sein
        assert np.all(matrix_strict == 0)


# ---------------------------------------------------------------------------
# VerificationResult – Properties
# ---------------------------------------------------------------------------

def _make_result(decision: str, kept: np.ndarray | None = None) -> VerificationResult:
    """Hilfsfunktion zum Erstellen eines VerificationResult mit beliebiger Entscheidung."""
    mat = np.array([[0, 1], [0, 0]], dtype=int)
    L = TransferFunction([1.0], [1.0, 1.0])
    return VerificationResult(
        decision=decision,
        matrix_A=mat.copy(),
        matrix_B=mat.copy(),
        kept_matrix=kept,
        open_loop_A=L,
        open_loop_B=L,
        label_A="A",
        label_B="B",
    )


class TestVerificationResultProperties:
    @pytest.mark.parametrize("decision", ["equal", "equal_keep_A", "equal_keep_B"])
    def test_are_equivalent_true(self, decision: str) -> None:
        assert _make_result(decision).are_equivalent is True

    @pytest.mark.parametrize("decision", ["keep_A", "keep_B", "keep_both"])
    def test_are_equivalent_false(self, decision: str) -> None:
        assert _make_result(decision).are_equivalent is False

    def test_a_contains_b_true(self) -> None:
        assert _make_result("keep_A").a_contains_b is True

    @pytest.mark.parametrize("decision", ["equal", "keep_B", "keep_both"])
    def test_a_contains_b_false(self, decision: str) -> None:
        assert _make_result(decision).a_contains_b is False

    def test_b_contains_a_true(self) -> None:
        assert _make_result("keep_B").b_contains_a is True

    @pytest.mark.parametrize("decision", ["equal", "keep_A", "keep_both"])
    def test_b_contains_a_false(self, decision: str) -> None:
        assert _make_result(decision).b_contains_a is False

    def test_are_independent_true(self) -> None:
        assert _make_result("keep_both").are_independent is True

    @pytest.mark.parametrize("decision", ["equal", "keep_A", "keep_B"])
    def test_are_independent_false(self, decision: str) -> None:
        assert _make_result(decision).are_independent is False


class TestVerificationResultSummary:
    def test_summary_equal(self) -> None:
        assert "äquivalent" in _make_result("equal").summary.lower()

    def test_summary_equal_keep_a(self) -> None:
        r = _make_result("equal_keep_A")
        assert "äquivalent" in r.summary.lower()
        assert "A" in r.summary

    def test_summary_equal_keep_b(self) -> None:
        r = _make_result("equal_keep_B")
        assert "äquivalent" in r.summary.lower()
        assert "B" in r.summary

    def test_summary_keep_a(self) -> None:
        r = _make_result("keep_A")
        assert "A" in r.summary

    def test_summary_keep_b(self) -> None:
        r = _make_result("keep_B")
        assert "B" in r.summary

    def test_summary_keep_both(self) -> None:
        assert "verschieden" in _make_result("keep_both").summary.lower()

    def test_summary_unknown_decision(self) -> None:
        r = _make_result("something_unexpected")
        assert "something_unexpected" in r.summary


# ---------------------------------------------------------------------------
# verify_loops
# ---------------------------------------------------------------------------

class TestVerifyLoops:
    """Tests für verify_loops – subgraph-Paket wird gemockt."""

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _run(decision: str, kept: np.ndarray | None,
             use_adj_list: bool = False) -> VerificationResult:
        """Führt verify_loops mit gemocktem subgraph-Modul aus."""
        G = TransferFunction([1.0], [1.0, 1.4, 1.0], name="G")
        C = TransferFunction([2.0], [1.0], name="C")
        mock_mod = _make_subgraph_mock(decision, kept)
        with patch.dict(sys.modules, {"subgraph": mock_mod}):
            return verify_loops(G, C, G, C, use_adjacency_list=use_adj_list)

    # -- import error ---------------------------------------------------

    def test_missing_subgraph_raises_import_error(self) -> None:
        G = TransferFunction([1.0], [1.0, 1.0])
        C = TransferFunction([1.0], [1.0])
        with patch.dict(sys.modules, {"subgraph": None}):
            with pytest.raises(ImportError, match="subgraph"):
                verify_loops(G, C, G, C)

    # -- decision branches --------------------------------------------

    def test_equal_decision(self) -> None:
        mat = np.array([[0, 1], [0, 0]], dtype=float)
        r = self._run("equal", mat)
        assert r.decision == "equal"
        assert r.are_equivalent

    def test_equal_keep_a_decision(self) -> None:
        mat = np.array([[0, 1], [0, 0]], dtype=float)
        r = self._run("equal_keep_A", mat)
        assert r.decision == "equal_keep_A"
        assert r.are_equivalent

    def test_equal_keep_b_decision(self) -> None:
        mat = np.array([[0, 1], [0, 0]], dtype=float)
        r = self._run("equal_keep_B", mat)
        assert r.decision == "equal_keep_B"
        assert r.are_equivalent

    def test_keep_b_decision(self) -> None:
        mat = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        r = self._run("keep_B", mat)
        assert r.decision == "keep_B"
        assert r.b_contains_a
        assert r.kept_matrix is not None

    def test_keep_a_decision(self) -> None:
        mat = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        r = self._run("keep_A", mat)
        assert r.decision == "keep_A"
        assert r.a_contains_b

    def test_keep_both_decision_kept_matrix_is_none(self) -> None:
        r = self._run("keep_both", None)
        assert r.decision == "keep_both"
        assert r.are_independent
        assert r.kept_matrix is None

    # -- adjacency-list variant ------------------------------------------

    def test_adjacency_list_variant(self) -> None:
        mat = np.array([[0, 1], [0, 0]], dtype=float)
        r = self._run("equal", mat, use_adj_list=True)
        assert r.are_equivalent

    # -- padding für unterschiedliche Systemordnungen --------------------

    def test_padding_different_orders(self) -> None:
        """Regelkreise unterschiedlicher Ordnung → Padding auf gleiche Größe."""
        G_small = TransferFunction([1.0], [1.0, 1.0])         # Ordnung 1
        G_large = TransferFunction([1.0], [1.0, 3.0, 3.0, 1.0])  # Ordnung 3
        C = TransferFunction([1.0], [1.0])

        kept_mat = np.zeros((3, 3), dtype=float)
        mock_mod = _make_subgraph_mock("keep_B", kept_mat)
        with patch.dict(sys.modules, {"subgraph": mock_mod}):
            r = verify_loops(G_small, C, G_large, C)

        assert r.decision == "keep_B"
        # Origial-Matrizen bleiben in ursprünglicher Größe
        assert r.matrix_A.shape[0] < r.matrix_B.shape[0]

    # -- labels ----------------------------------------------------------

    def test_custom_labels_stored(self) -> None:
        G = TransferFunction([1.0], [1.0, 1.0])
        C = TransferFunction([1.0], [1.0])
        mock_mod = _make_subgraph_mock("equal", np.zeros((1, 1), dtype=float))
        with patch.dict(sys.modules, {"subgraph": mock_mod}):
            r = verify_loops(G, C, G, C, label_A="Strecke X", label_B="Strecke Y")
        assert r.label_A == "Strecke X"
        assert r.label_B == "Strecke Y"

    # -- open-loop TFs in result -----------------------------------------

    def test_result_contains_open_loop_tfs(self) -> None:
        G = TransferFunction([1.0], [1.0, 1.0])
        C = TransferFunction([1.0], [1.0])
        mock_mod = _make_subgraph_mock("equal", np.zeros((1, 1), dtype=float))
        with patch.dict(sys.modules, {"subgraph": mock_mod}):
            r = verify_loops(G, C, G, C)
        assert isinstance(r.open_loop_A, TransferFunction)
        assert isinstance(r.open_loop_B, TransferFunction)

    # -- matrix dtype (int) from kept_matrix ----------------------------

    def test_kept_matrix_dtype_is_int(self) -> None:
        mat = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
        r = self._run("equal", mat)
        assert np.issubdtype(r.kept_matrix.dtype, np.integer)

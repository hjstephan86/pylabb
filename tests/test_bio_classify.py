"""
tests/test_bio_classify.py
==========================
100 % Testabdeckung für pylabb.control.bio_classify.

Geprüfte Einheiten
------------------
* BioClass              – Enum-Werte (class_id, label)
* BioExtension          – Datenklasse
* BioClassResult        – alle Properties (is_classified, summary) inkl.
                          UNCLASSIFIED-Zweig
* _has_integrator       – Pol bei s=0 / kein Pol bei s=0
* _has_derivative       – unecht / echt gebrochen
* _pad_matrix           – n >= target (Kopie), n < target (Padding)
* _matrix_contains_reference – R leer, M kleiner als R, volle Überlappung,
                          Teilüberlappung
* _compute_subgraph_chain – UNCLASSIFIED, MIMOSE, alle linearen Klassen
* _classify_by_rules    – alle sieben Entscheidungszweige
* get_reference_topology – alle sieben Klassen + KeyError bei UNCLASSIFIED
* get_extensions        – alle acht Klassen (inkl. UNCLASSIFIED → leer)
* classify_loop         – alle sieben biologischen Klassen, hysteresis_band-Pfad,
                          UNCLASSIFIED-Konfidenzpfad, Subgraph-Kette,
                          BioClassResult-Felder
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.bio_classify import (
    BioClass,
    BioClassResult,
    BioExtension,
    _classify_by_rules,
    _compute_subgraph_chain,
    _has_derivative,
    _has_integrator,
    _matrix_contains_reference,
    _pad_matrix,
    classify_loop,
    get_extensions,
    get_reference_topology,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def pt1() -> TransferFunction:
    """PT1-Strecke G(s) = 1 / (s + 1)."""
    return TransferFunction([1.0], [1.0, 1.0], name="PT1")


@pytest.fixture()
def pt2() -> TransferFunction:
    """PT2-Strecke G(s) = 1 / (s² + 1.4s + 1)."""
    return TransferFunction([1.0], [1.0, 1.4, 1.0], name="PT2")


@pytest.fixture()
def p_ctrl() -> TransferFunction:
    """P-Regler C(s) = 2."""
    return TransferFunction([2.0], [1.0], name="C_P")


@pytest.fixture()
def pi_ctrl() -> TransferFunction:
    """PI-Regler C(s) = (s + 1) / s."""
    return TransferFunction([1.0, 1.0], [1.0, 0.0], name="C_PI")


@pytest.fixture()
def pid_ctrl() -> TransferFunction:
    """PID-Regler C(s) = (s² + s + 1) / s (vereinfacht)."""
    return TransferFunction([1.0, 1.0, 1.0], [1.0, 0.0], name="C_PID")


# ===========================================================================
# BioClass – Enum
# ===========================================================================

class TestBioClassEnum:
    def test_class_ids_unique(self) -> None:
        ids = [m.class_id for m in BioClass]
        assert len(ids) == len(set(ids))

    def test_labels_not_empty(self) -> None:
        for member in BioClass:
            assert len(member.label) > 0

    def test_known_members(self) -> None:
        assert BioClass.SEESTERN.class_id == 1
        assert BioClass.PUPILLE.class_id == 2
        assert BioClass.KLEINHIRN.class_id == 3
        assert BioClass.HERZ.class_id == 4
        assert BioClass.ALBATROS.class_id == 5
        assert BioClass.MIMOSE.class_id == 6
        assert BioClass.TINTENFISCH.class_id == 7
        assert BioClass.UNCLASSIFIED.class_id == 0

    def test_labels(self) -> None:
        assert BioClass.SEESTERN.label == "Seestern"
        assert BioClass.PUPILLE.label == "Pupille"
        assert BioClass.UNCLASSIFIED.label == "Unklassifiziert"


# ===========================================================================
# BioExtension – Datenklasse
# ===========================================================================

class TestBioExtension:
    def test_default_code_hint_empty(self) -> None:
        ext = BioExtension(name="Test", description="Beschreibung")
        assert ext.code_hint == ""

    def test_with_code_hint(self) -> None:
        ext = BioExtension(name="T", description="D", code_hint="foo()")
        assert ext.code_hint == "foo()"

    def test_fields_accessible(self) -> None:
        ext = BioExtension(name="N", description="D")
        assert ext.name == "N"
        assert ext.description == "D"


# ===========================================================================
# BioClassResult – Properties
# ===========================================================================

class TestBioClassResult:
    def _make_result(self, bio_class: BioClass) -> BioClassResult:
        """Erzeugt ein minimales BioClassResult für Tests."""
        return BioClassResult(
            bio_class=bio_class,
            confidence=0.9,
            subgraph_chain=_compute_subgraph_chain(bio_class),
            adjacency_matrix=np.array([[0, 1], [1, 0]], dtype=int),
            density=0.5,
            has_self_loop=False,
            has_full_last_row=True,
            controller_has_integrator=False,
            controller_has_derivative=False,
        )

    # -- is_classified -------------------------------------------------------

    def test_is_classified_true(self) -> None:
        result = self._make_result(BioClass.SEESTERN)
        assert result.is_classified is True  # line 215

    def test_is_classified_false_for_unclassified(self) -> None:
        result = self._make_result(BioClass.UNCLASSIFIED)
        assert result.is_classified is False

    # -- summary -------------------------------------------------------------

    def test_summary_unclassified(self) -> None:  # lines 220-223
        result = self._make_result(BioClass.UNCLASSIFIED)
        s = result.summary
        assert "keiner biologischen Äquivalenzklasse" in s

    def test_summary_classified_contains_class_id(self) -> None:  # lines 224-230
        result = self._make_result(BioClass.PUPILLE)
        s = result.summary
        assert "2" in s
        assert "Pupille" in s
        assert "%" in s  # Konfidenz-Zeile

    def test_summary_classified_contains_chain(self) -> None:
        result = self._make_result(BioClass.KLEINHIRN)
        s = result.summary
        # Kette Seestern ⊆ Pupille ⊆ Kleinhirn
        assert "Seestern" in s
        assert "Kleinhirn" in s


# ===========================================================================
# _has_integrator
# ===========================================================================

class TestHasIntegrator:
    def test_pure_integrator(self) -> None:  # lines 250-251
        tf_i = TransferFunction([1.0], [1.0, 0.0])  # 1/s
        assert _has_integrator(tf_i) is True

    def test_pi_controller(self) -> None:
        pi = TransferFunction([1.0, 1.0], [1.0, 0.0])  # (s+1)/s
        assert _has_integrator(pi) is True

    def test_p_controller_no_integrator(self) -> None:
        p = TransferFunction([2.0], [1.0])  # C(s)=2
        assert _has_integrator(p) is False

    def test_pt1_no_integrator(self, pt1: TransferFunction) -> None:
        assert _has_integrator(pt1) is False

    def test_threshold_boundary(self) -> None:
        """Nenner-Konstante exakt bei threshold: sollte als Integrator erkannt werden."""
        # Den[-1] = 5e-7 < 1e-6 → True
        tf = TransferFunction([1.0], [1.0, 5e-7])
        assert _has_integrator(tf, threshold=1e-6) is True

    def test_above_threshold(self) -> None:
        """Nenner-Konstante größer threshold: kein Integrator."""
        tf = TransferFunction([1.0], [1.0, 1.0])
        assert _has_integrator(tf, threshold=1e-6) is False


# ===========================================================================
# _has_derivative
# ===========================================================================

class TestHasDerivative:
    def test_proper_no_derivative(self, pt1: TransferFunction) -> None:  # line 269
        assert _has_derivative(pt1) is False

    def test_improper_has_derivative(self) -> None:
        # Zählergrad > Nennergrad: C(s) = s²/(s+1)
        tf = TransferFunction([1.0, 0.0, 0.0], [1.0, 1.0])
        assert _has_derivative(tf) is True

    def test_biproper_no_derivative(self) -> None:
        # Grad(Zähler) == Grad(Nenner): biproper, kein D
        tf = TransferFunction([1.0, 1.0], [1.0, 2.0])
        assert _has_derivative(tf) is False

    def test_pid_with_derivative(self) -> None:
        # Typischer PID: (Kd s² + Kp s + Ki) / s → Zählergrad 2 > Nennergrad 1
        pid = TransferFunction([0.5, 2.0, 1.0], [1.0, 0.0])
        assert _has_derivative(pid) is True


# ===========================================================================
# _pad_matrix
# ===========================================================================

class TestPadMatrix:
    def test_no_padding_needed(self) -> None:  # branch n >= target_n
        M = np.array([[1, 2], [3, 4]])
        result = _pad_matrix(M, target_n=2)
        assert result.shape == (2, 2)
        assert np.array_equal(result, M)
        # Muss eine Kopie sein
        result[0, 0] = 99
        assert M[0, 0] == 1

    def test_target_smaller_than_n(self) -> None:
        """target_n < n: Matrix unverändert zurückgeben."""
        M = np.eye(4, dtype=int)
        result = _pad_matrix(M, target_n=2)
        assert result.shape == (4, 4)

    def test_padding_increases_size(self) -> None:  # lines 284-289
        M = np.ones((2, 2), dtype=int)
        result = _pad_matrix(M, target_n=4)
        assert result.shape == (4, 4)
        # Obere linke 2×2 bleibt erhalten
        assert np.array_equal(result[:2, :2], M)
        # Padding-Bereich ist Null
        assert np.all(result[2:, :] == 0)
        assert np.all(result[:, 2:] == 0)

    def test_padding_preserves_dtype(self) -> None:
        M = np.array([[1, 0], [0, 1]], dtype=np.int32)
        result = _pad_matrix(M, target_n=3)
        assert result.dtype == np.int32


# ===========================================================================
# _matrix_contains_reference
# ===========================================================================

class TestMatrixContainsReference:
    def test_empty_reference_returns_one(self) -> None:  # line 314 (ref_edges==0)
        M = np.ones((2, 2), dtype=int)
        R = np.zeros((2, 2), dtype=int)
        assert _matrix_contains_reference(M, R) == 1.0

    def test_full_overlap(self) -> None:  # lines 310-321
        M = np.ones((3, 3), dtype=int)
        R = np.ones((3, 3), dtype=int)
        conf = _matrix_contains_reference(M, R)
        assert conf == pytest.approx(1.0)

    def test_zero_overlap(self) -> None:
        M = np.zeros((3, 3), dtype=int)
        R = np.ones((3, 3), dtype=int)
        conf = _matrix_contains_reference(M, R)
        assert conf == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        M = np.array([[1, 0], [0, 1]], dtype=int)  # 2 Kanten
        R = np.ones((2, 2), dtype=int)             # 4 Kanten
        conf = _matrix_contains_reference(M, R)
        assert conf == pytest.approx(2 / 4)

    def test_m_smaller_than_r_padded(self) -> None:
        """M kleiner als R: M wird gepaddet."""
        M = np.array([[1, 1], [1, 1]], dtype=int)   # 2×2
        R = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=int)  # 3×3, 4 Kanten
        conf = _matrix_contains_reference(M, R)
        assert conf == pytest.approx(1.0)  # alle 4 Referenzkanten in gepaddetem M

    def test_r_larger_than_m_partial(self) -> None:
        """R größer als M: nur Überlappung in oberer linker Ecke."""
        M = np.array([[0, 1], [1, 0]], dtype=int)   # 2×2, 2 Kanten
        R = np.ones((4, 4), dtype=int)              # 4×4, 16 Kanten
        conf = _matrix_contains_reference(M, R)
        # Im gepadeten M: nur M-Bereich hat Einsen → 2 von 16 gefunden
        assert conf == pytest.approx(2 / 16)


# ===========================================================================
# _compute_subgraph_chain
# ===========================================================================

class TestComputeSubgraphChain:
    def test_unclassified_returns_empty(self) -> None:  # line 341
        chain = _compute_subgraph_chain(BioClass.UNCLASSIFIED)
        assert chain == []

    def test_mimose_returns_seestern_mimose(self) -> None:  # lines 342-343
        chain = _compute_subgraph_chain(BioClass.MIMOSE)
        assert chain == [BioClass.SEESTERN, BioClass.MIMOSE]

    def test_seestern_chain(self) -> None:  # lines 344-346
        chain = _compute_subgraph_chain(BioClass.SEESTERN)
        assert chain == [BioClass.SEESTERN]

    def test_pupille_chain(self) -> None:
        chain = _compute_subgraph_chain(BioClass.PUPILLE)
        assert chain == [BioClass.SEESTERN, BioClass.PUPILLE]

    def test_kleinhirn_chain(self) -> None:
        chain = _compute_subgraph_chain(BioClass.KLEINHIRN)
        assert chain == [BioClass.SEESTERN, BioClass.PUPILLE, BioClass.KLEINHIRN]

    def test_herz_chain(self) -> None:
        chain = _compute_subgraph_chain(BioClass.HERZ)
        assert chain[-1] == BioClass.HERZ
        assert BioClass.SEESTERN in chain

    def test_albatros_chain(self) -> None:
        chain = _compute_subgraph_chain(BioClass.ALBATROS)
        assert chain[-1] == BioClass.ALBATROS
        assert len(chain) == 5

    def test_tintenfisch_chain_is_full(self) -> None:
        chain = _compute_subgraph_chain(BioClass.TINTENFISCH)
        assert chain[-1] == BioClass.TINTENFISCH
        assert len(chain) == 6  # alle linearen Klassen


# ===========================================================================
# _classify_by_rules
# ===========================================================================

BASE_KWARGS = dict(
    density=0.3,
    has_self_loop=False,
    has_full_last_row=False,
    c_has_integrator=False,
    c_has_derivative=False,
    has_feedforward=False,
    has_cascade=False,
    is_full_state_feedback=False,
    is_mimo=False,
    has_hysteresis=False,
    has_disturbance_channel=False,
)


class TestClassifyByRules:
    """Deckt alle Entscheidungszweige von _classify_by_rules ab."""

    def test_seestern_all_false(self) -> None:  # lines 418 (Seestern-Rückgabe)
        assert _classify_by_rules(**BASE_KWARGS) == BioClass.SEESTERN

    def test_pupille_via_integrator(self) -> None:  # lines 414-415
        kw = {**BASE_KWARGS, "c_has_integrator": True}
        assert _classify_by_rules(**kw) == BioClass.PUPILLE

    def test_mimose_via_hysteresis(self) -> None:  # lines 410-411
        kw = {**BASE_KWARGS, "has_hysteresis": True}
        assert _classify_by_rules(**kw) == BioClass.MIMOSE

    def test_kleinhirn_via_feedforward(self) -> None:  # lines 407-408
        kw = {**BASE_KWARGS, "has_feedforward": True}
        assert _classify_by_rules(**kw) == BioClass.KLEINHIRN

    def test_herz_via_cascade(self) -> None:  # lines 403-404
        kw = {**BASE_KWARGS, "has_cascade": True}
        assert _classify_by_rules(**kw) == BioClass.HERZ

    def test_albatros_via_full_state_feedback(self) -> None:  # lines 398-399
        kw = {**BASE_KWARGS, "is_full_state_feedback": True}
        assert _classify_by_rules(**kw) == BioClass.ALBATROS

    def test_albatros_via_full_last_row_and_disturbance(self) -> None:  # lines 398-399
        kw = {**BASE_KWARGS, "has_full_last_row": True, "has_disturbance_channel": True}
        assert _classify_by_rules(**kw) == BioClass.ALBATROS

    def test_tintenfisch_via_mimo(self) -> None:  # lines 392-393
        kw = {**BASE_KWARGS, "is_mimo": True}
        assert _classify_by_rules(**kw) == BioClass.TINTENFISCH

    def test_tintenfisch_via_high_density(self) -> None:  # lines 392-393
        kw = {**BASE_KWARGS, "density": 0.8}
        assert _classify_by_rules(**kw) == BioClass.TINTENFISCH

    def test_tintenfisch_takes_priority_over_cascade(self) -> None:
        """Tintenfisch wird erkannt, auch wenn Kaskade gesetzt ist."""
        kw = {**BASE_KWARGS, "is_mimo": True, "has_cascade": True}
        assert _classify_by_rules(**kw) == BioClass.TINTENFISCH

    def test_albatros_takes_priority_over_cascade(self) -> None:
        kw = {**BASE_KWARGS, "is_full_state_feedback": True, "has_cascade": True}
        assert _classify_by_rules(**kw) == BioClass.ALBATROS

    def test_kleinhirn_takes_priority_over_pupille(self) -> None:
        kw = {**BASE_KWARGS, "has_feedforward": True, "c_has_integrator": True}
        assert _classify_by_rules(**kw) == BioClass.KLEINHIRN

    def test_mimose_takes_priority_over_pupille(self) -> None:
        kw = {**BASE_KWARGS, "has_hysteresis": True, "c_has_integrator": True}
        assert _classify_by_rules(**kw) == BioClass.MIMOSE


# ===========================================================================
# get_reference_topology
# ===========================================================================

class TestGetReferenceTopology:
    def test_returns_numpy_array(self) -> None:  # line 644
        R = get_reference_topology(BioClass.SEESTERN)
        assert isinstance(R, np.ndarray)

    def test_returns_copy(self) -> None:
        """Veränderungen dürfen die interne Topologie nicht zerstören."""
        R1 = get_reference_topology(BioClass.ALBATROS)
        R1[0, 0] = 99
        R2 = get_reference_topology(BioClass.ALBATROS)
        assert R2[0, 0] != 99

    def test_all_seven_classes_have_topology(self) -> None:
        classified = [
            BioClass.SEESTERN, BioClass.PUPILLE, BioClass.KLEINHIRN,
            BioClass.HERZ, BioClass.ALBATROS, BioClass.MIMOSE, BioClass.TINTENFISCH,
        ]
        for c in classified:
            R = get_reference_topology(c)
            assert R.ndim == 2
            assert R.shape[0] == R.shape[1]
            assert R.shape[0] >= 2

    def test_unclassified_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_reference_topology(BioClass.UNCLASSIFIED)

    def test_albatros_last_row_all_ones(self) -> None:
        """Albatros-Referenz hat eine vollständig besetzte letzte Zeile (LQR)."""
        R = get_reference_topology(BioClass.ALBATROS)
        assert np.all(R[-1, :] == 1)

    def test_tintenfisch_is_all_ones(self) -> None:
        """Tintenfisch-Referenz ist die vollständige Matrix."""
        R = get_reference_topology(BioClass.TINTENFISCH)
        assert np.all(R == 1)

    def test_pupille_has_self_loop(self) -> None:
        """Pupillen-Referenz hat Diagonaleintrag (Integralknoten)."""
        R = get_reference_topology(BioClass.PUPILLE)
        assert R[0, 0] == 1


# ===========================================================================
# get_extensions
# ===========================================================================

class TestGetExtensions:
    def test_unclassified_returns_empty(self) -> None:  # line 658
        exts = get_extensions(BioClass.UNCLASSIFIED)
        assert exts == []

    def test_returns_list(self) -> None:
        exts = get_extensions(BioClass.SEESTERN)
        assert isinstance(exts, list)

    def test_all_classes_have_extensions(self) -> None:
        classified = [
            BioClass.SEESTERN, BioClass.PUPILLE, BioClass.KLEINHIRN,
            BioClass.HERZ, BioClass.ALBATROS, BioClass.MIMOSE, BioClass.TINTENFISCH,
        ]
        for c in classified:
            exts = get_extensions(c)
            assert len(exts) > 0, f"{c} sollte Erweiterungen haben"

    def test_returns_bio_extension_objects(self) -> None:
        exts = get_extensions(BioClass.PUPILLE)
        for ext in exts:
            assert isinstance(ext, BioExtension)
            assert len(ext.name) > 0
            assert len(ext.description) > 0

    def test_returns_new_list_each_time(self) -> None:
        """Änderungen an der Rückgabe dürfen den internen Zustand nicht beeinflussen."""
        exts1 = get_extensions(BioClass.SEESTERN)
        exts1.clear()
        exts2 = get_extensions(BioClass.SEESTERN)
        assert len(exts2) > 0

    def test_anti_windup_in_pupille(self) -> None:
        exts = get_extensions(BioClass.PUPILLE)
        names = [e.name for e in exts]
        assert any("Anti-Windup" in n for n in names)

    def test_ilc_in_kleinhirn(self) -> None:
        exts = get_extensions(BioClass.KLEINHIRN)
        names = [e.name for e in exts]
        assert any("ILC" in n or "Learning" in n for n in names)

    def test_mpc_in_albatros(self) -> None:
        exts = get_extensions(BioClass.ALBATROS)
        names = [e.name for e in exts]
        assert any("MPC" in n for n in names)


# ===========================================================================
# classify_loop – Integration
# ===========================================================================

class TestClassifyLoop:
    """Testet classify_loop mit realen TransferFunction-Objekten."""

    # ------------------------------------------------------------------
    # Klasse I – Seestern
    # ------------------------------------------------------------------

    def test_seestern_p_controller(self, pt2: TransferFunction, p_ctrl: TransferFunction) -> None:
        """P-Regler an PT2-Strecke → Seestern."""  # lines 722-775
        result = classify_loop(pt2, p_ctrl)
        assert result.bio_class == BioClass.SEESTERN
        assert result.is_classified is True
        assert 0.0 <= result.confidence <= 1.0
        assert result.subgraph_chain == [BioClass.SEESTERN]
        assert result.adjacency_matrix.ndim == 2
        assert isinstance(result.density, float)
        assert isinstance(result.has_self_loop, bool)
        assert isinstance(result.has_full_last_row, bool)
        assert result.controller_has_integrator is False

    def test_seestern_pt2_p(self, pt2: TransferFunction, p_ctrl: TransferFunction) -> None:
        result = classify_loop(pt2, p_ctrl)
        assert result.bio_class == BioClass.SEESTERN

    # ------------------------------------------------------------------
    # Klasse II – Pupille
    # ------------------------------------------------------------------

    def test_pupille_pi_controller(self, pt2: TransferFunction, pi_ctrl: TransferFunction) -> None:
        """PI-Regler → Pupille."""
        result = classify_loop(pt2, pi_ctrl)
        assert result.bio_class == BioClass.PUPILLE
        assert result.controller_has_integrator is True
        assert BioClass.SEESTERN in result.subgraph_chain
        assert BioClass.PUPILLE in result.subgraph_chain

    def test_pupille_pt1_pi(self, pt1: TransferFunction, pi_ctrl: TransferFunction) -> None:
        result = classify_loop(pt1, pi_ctrl)
        assert result.bio_class == BioClass.PUPILLE

    # ------------------------------------------------------------------
    # Klasse III – Kleinhirn
    # ------------------------------------------------------------------

    def test_kleinhirn_feedforward(self, pt2: TransferFunction, pi_ctrl: TransferFunction) -> None:
        """PI + Vorsteuerung → Kleinhirn."""
        result = classify_loop(pt2, pi_ctrl, has_feedforward=True)
        assert result.bio_class == BioClass.KLEINHIRN
        assert BioClass.KLEINHIRN in result.subgraph_chain

    def test_kleinhirn_subgraph_chain_contains_pupille(
        self, pt2: TransferFunction, pi_ctrl: TransferFunction
    ) -> None:
        result = classify_loop(pt2, pi_ctrl, has_feedforward=True)
        chain = result.subgraph_chain
        assert BioClass.SEESTERN in chain
        assert BioClass.PUPILLE in chain

    # ------------------------------------------------------------------
    # Klasse IV – Herz
    # ------------------------------------------------------------------

    def test_herz_cascade(self, pt2: TransferFunction, p_ctrl: TransferFunction) -> None:
        """Kaskadenstruktur → Herz."""
        result = classify_loop(pt2, p_ctrl, has_cascade=True)
        assert result.bio_class == BioClass.HERZ
        assert BioClass.HERZ in result.subgraph_chain

    # ------------------------------------------------------------------
    # Klasse V – Albatros
    # ------------------------------------------------------------------

    def test_albatros_full_state_feedback(
        self, pt2: TransferFunction, p_ctrl: TransferFunction
    ) -> None:
        """LQR → Albatros."""
        result = classify_loop(pt2, p_ctrl, is_full_state_feedback=True)
        assert result.bio_class == BioClass.ALBATROS
        assert BioClass.ALBATROS in result.subgraph_chain

    def test_albatros_disturbance_channel(
        self, pt2: TransferFunction, p_ctrl: TransferFunction
    ) -> None:
        """Vollständige letzte Zeile + Störkanal → Albatros (falls Matrix zutrifft)."""
        result = classify_loop(
            pt2, p_ctrl,
            is_full_state_feedback=True,
            has_disturbance_channel=True,
        )
        assert result.bio_class == BioClass.ALBATROS

    # ------------------------------------------------------------------
    # Klasse VI – Mimose
    # ------------------------------------------------------------------

    def test_mimose_hysteresis_flag(
        self, pt2: TransferFunction, p_ctrl: TransferFunction
    ) -> None:
        """has_hysteresis=True → Mimose."""
        result = classify_loop(pt2, p_ctrl, has_hysteresis=True)
        assert result.bio_class == BioClass.MIMOSE
        assert result.subgraph_chain == [BioClass.SEESTERN, BioClass.MIMOSE]

    def test_mimose_hysteresis_band(
        self, pt2: TransferFunction, p_ctrl: TransferFunction
    ) -> None:
        """hysteresis_band > 0 → aktiviert Hysterese-Pfad."""
        result = classify_loop(pt2, p_ctrl, hysteresis_band=0.5)
        assert result.bio_class == BioClass.MIMOSE

    def test_mimose_hysteresis_band_zero_not_mimose(
        self, pt2: TransferFunction, p_ctrl: TransferFunction
    ) -> None:
        """hysteresis_band == 0 und has_hysteresis == False → kein Mimose."""
        result = classify_loop(pt2, p_ctrl, hysteresis_band=0.0)
        assert result.bio_class != BioClass.MIMOSE

    # ------------------------------------------------------------------
    # UNCLASSIFIED-Konfidenzpfad (line 762 in bio_classify.py)
    # ------------------------------------------------------------------

    def test_unclassified_confidence_zero(
        self, pt2: TransferFunction, p_ctrl: TransferFunction
    ) -> None:
        """Wenn _classify_by_rules UNCLASSIFIED zurückgibt, ist confidence=0.0."""
        with patch(
            "pylabb.control.bio_classify._classify_by_rules",
            return_value=BioClass.UNCLASSIFIED,
        ):
            result = classify_loop(pt2, p_ctrl)
        assert result.bio_class is BioClass.UNCLASSIFIED
        assert result.confidence == 0.0
        assert result.is_classified is False
        assert result.subgraph_chain == []

    # ------------------------------------------------------------------
    # Klasse VII – Tintenfisch
    # ------------------------------------------------------------------

    def test_tintenfisch_is_mimo(
        self, pt1: TransferFunction, p_ctrl: TransferFunction
    ) -> None:
        """is_mimo=True → Tintenfisch."""
        result = classify_loop(pt1, p_ctrl, is_mimo=True)
        assert result.bio_class == BioClass.TINTENFISCH
        assert result.subgraph_chain[-1] == BioClass.TINTENFISCH

    # ------------------------------------------------------------------
    # confidence & Konfidenz-Pfad
    # ------------------------------------------------------------------

    def test_confidence_in_range(self, pt2: TransferFunction, pi_ctrl: TransferFunction) -> None:
        result = classify_loop(pt2, pi_ctrl)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_seestern(self, pt1: TransferFunction, p_ctrl: TransferFunction) -> None:
        result = classify_loop(pt1, p_ctrl)
        assert result.confidence > 0.0

    def test_confidence_does_not_exceed_one(
        self, pt2: TransferFunction, p_ctrl: TransferFunction
    ) -> None:
        """min(1.0, raw_conf + 0.05) darf 1.0 nicht überschreiten."""
        result = classify_loop(pt2, p_ctrl)
        assert result.confidence <= 1.0

    # ------------------------------------------------------------------
    # summary – integrativ über classify_loop
    # ------------------------------------------------------------------

    def test_summary_contains_bio_class_label(
        self, pt2: TransferFunction, pi_ctrl: TransferFunction
    ) -> None:
        result = classify_loop(pt2, pi_ctrl)
        assert "Pupille" in result.summary

    # ------------------------------------------------------------------
    # Result-Felder vollständig belegt
    # ------------------------------------------------------------------

    def test_result_fields_set(self, pt2: TransferFunction, pi_ctrl: TransferFunction) -> None:
        result = classify_loop(pt2, pi_ctrl)
        assert result.adjacency_matrix is not None
        assert result.adjacency_matrix.ndim == 2
        assert 0.0 <= result.density <= 1.0
        assert isinstance(result.has_self_loop, bool)
        assert isinstance(result.has_full_last_row, bool)
        assert isinstance(result.controller_has_integrator, bool)
        assert isinstance(result.controller_has_derivative, bool)

    def test_derivative_flag_set_for_improper_controller(
        self, pt2: TransferFunction
    ) -> None:
        # PID-Näherung: Zählergrad > Nennergrad
        pid = TransferFunction([1.0, 2.0, 1.0], [1.0, 0.0])
        result = classify_loop(pt2, pid)
        assert result.controller_has_derivative is True

    def test_no_derivative_for_pi_controller(
        self, pt2: TransferFunction, pi_ctrl: TransferFunction
    ) -> None:
        result = classify_loop(pt2, pi_ctrl)
        assert result.controller_has_derivative is False

    # ------------------------------------------------------------------
    # threshold-Parameter
    # ------------------------------------------------------------------

    def test_custom_threshold(self, pt1: TransferFunction, p_ctrl: TransferFunction) -> None:
        result = classify_loop(pt1, p_ctrl, threshold=1e-6)
        assert isinstance(result, BioClassResult)

    # ------------------------------------------------------------------
    # get_extensions über classify_loop-Ergebnis
    # ------------------------------------------------------------------

    def test_get_extensions_after_classify(
        self, pt2: TransferFunction, pi_ctrl: TransferFunction
    ) -> None:
        result = classify_loop(pt2, pi_ctrl)
        exts = get_extensions(result.bio_class)
        assert len(exts) > 0

    # ------------------------------------------------------------------
    # Import aus pylabb.control (via __init__.py)
    # ------------------------------------------------------------------

    def test_import_via_control_package(self) -> None:
        from pylabb.control import (
            BioClass as BC,
            BioClassResult as BCR,
            BioExtension as BE,
            classify_loop as cl,
            get_extensions as ge,
            get_reference_topology as grt,
        )
        assert BC.SEESTERN.class_id == 1
        assert issubclass(BCR, object)
        assert issubclass(BE, object)
        assert callable(cl)
        assert callable(ge)
        assert callable(grt)

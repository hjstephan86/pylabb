"""
pylabb.control.verification
============================
Äquivalenzüberprüfung zweier Regelkreise mittels Subgraph Algorithmus.

Der geschlossene Regelkreis (Strecke G(s) + Regler C(s)) wird in einen
gerichteten Signalflussgraphen überführt. Die daraus ableitbare
Adjazenzmatrix wird dem ``Subgraph``-Algorithmus übergeben, der prüft,
ob einer der beiden Graphen strukturell im anderen enthalten ist.

Graph-Modell
------------
Jeder Pol des offenen Kreises L(s) = C(s)·G(s) repräsentiert einen
Zustandsknoten im Signalflussgraphen. Kanten entstehen dort, wo die
Systemmatrix **A** des Zustandsraummodells von L(s) einen von null
verschiedenen Eintrag hat. Die resultierende Boole'sche Adjazenzmatrix
kodiert also die innere Kopplungsstruktur des Regelkreises.

Rückgabe-Entscheidungen des Subgraph Algorithmus
-------------------------------------------------
* ``"equal"``        – Beide Regelkreise sind graphenstrukturell äquivalent.
* ``"equal_keep_A"`` – Graphen sind äquivalent; A hat mehr oder gleich viele Kanten.
* ``"equal_keep_B"`` – Graphen sind äquivalent; B hat mehr Kanten (redundante Info).
* ``"keep_A"``       – G_B ist in G_A enthalten → Regelkreis A ist komplexer (übergeordnet).
* ``"keep_B"``       – G_A ist in G_B enthalten → Regelkreis B ist komplexer (übergeordnet).
* ``"keep_both"``    – Keine strukturelle Inklusion; Regelkreise sind verschieden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import scipy.signal as sp_sig

from pylabb.core.transfer_function import TransferFunction


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Regelkreis → Adjazenzmatrix
# ---------------------------------------------------------------------------

def _open_loop(plant: TransferFunction, controller: TransferFunction) -> TransferFunction:
    """Berechnet die Reihenschaltung L(s) = C(s)·G(s)."""
    num = np.polymul(controller.num, plant.num)
    den = np.polymul(controller.den, plant.den)
    return TransferFunction(num, den, name="L")


def _to_adjacency_matrix(tf: TransferFunction, threshold: float = 1e-9) -> NDArray[np.int_]:
    """
    Konvertiert eine Übertragungsfunktion in eine binäre Adjazenzmatrix.

    Vorgehensweise:
    1. Zustandsraumdarstellung berechnen (scipy, Observable Canonical Form).
    2. Systemmatrix **A** extrahieren.
    3. Einträge mit |A_ij| > threshold → 1, sonst → 0.

    Parameters
    ----------
    tf : Übertragungsfunktion (kontinuierlich).
    threshold : Schwellwert für signifikante Einträge (Standard: 1e-9).

    Returns
    -------
    Binäre Adjazenzmatrix (n × n), n = Ordnung von tf.
    """
    # scipy tf2ss: Observable Canonical Form → liefert A, B, C, D direkt
    A, _B, _C, _D = sp_sig.tf2ss(tf.num.real, tf.den.real)

    if A.size == 0 or A.ndim < 2:
        # Ordnung 0: 1×1 Matrix mit Eins (triviale Verbindung)
        return np.array([[1]], dtype=int)

    adj = (np.abs(A) > threshold).astype(int)
    return adj


def loop_to_graph(
    plant: TransferFunction,
    controller: TransferFunction,
    threshold: float = 1e-9,
) -> Tuple[NDArray[np.int_], TransferFunction]:
    """
    Wandelt einen Regelkreis in seinen Signalflussgraphen um.

    Parameters
    ----------
    plant : Strecken-Übertragungsfunktion G(s).
    controller : Regler-Übertragungsfunktion C(s).
    threshold : Schwellwert für Adjazenz-Einträge.

    Returns
    -------
    (adj_matrix, open_loop_tf) : Adjazenzmatrix und die verwendete
                                  offene Kreisübertragungsfunktion.
    """
    L = _open_loop(plant, controller)
    adj = _to_adjacency_matrix(L, threshold)
    return adj, L


# ---------------------------------------------------------------------------
# Verifikations-Ergebnis
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Ergebnis der Regelkreis-Äquivalenzüberprüfung.

    Attributes
    ----------
    decision : Entscheidung des Subgraph Algorithmus (vgl. Modul-Docstring).
    matrix_A : Adjazenzmatrix von Regelkreis A.
    matrix_B : Adjazenzmatrix von Regelkreis B.
    kept_matrix : Vom Algorithmus bevorzugte Matrix (oder ``None`` bei keep_both).
    open_loop_A : Offene Kreisübertragungsfunktion L_A(s).
    open_loop_B : Offene Kreisübertragungsfunktion L_B(s).
    label_A : Bezeichner für Regelkreis A.
    label_B : Bezeichner für Regelkreis B.
    """
    decision: str
    matrix_A: NDArray[np.int_]
    matrix_B: NDArray[np.int_]
    kept_matrix: Optional[NDArray[np.int_]]
    open_loop_A: TransferFunction
    open_loop_B: TransferFunction
    label_A: str = "Regelkreis A"
    label_B: str = "Regelkreis B"

    # ------------------------------------------------------------------
    # Bequemlichkeits-Properties
    # ------------------------------------------------------------------

    @property
    def are_equivalent(self) -> bool:
        """True, wenn beide Strukturen graphenäquivalent sind."""
        return self.decision.startswith("equal")

    @property
    def a_contains_b(self) -> bool:
        """True, wenn Graph A den Graph B enthält (A ist komplexer)."""
        return self.decision == "keep_A"

    @property
    def b_contains_a(self) -> bool:
        """True, wenn Graph B den Graph A enthält (B ist komplexer)."""
        return self.decision == "keep_B"

    @property
    def are_independent(self) -> bool:
        """True, wenn keine strukturelle Inklusion vorliegt."""
        return self.decision == "keep_both"

    @property
    def summary(self) -> str:
        """Lesbare Zusammenfassung der Verifikation."""
        mapping = {
            "equal": "Regelkreise sind graphenstrukturell äquivalent.",
            "equal_keep_A": (
                "Regelkreise sind äquivalent; "
                f"{self.label_A} hat mehr Kanten (bevorzugt)."
            ),
            "equal_keep_B": (
                "Regelkreise sind äquivalent; "
                f"{self.label_B} hat mehr Kanten (bevorzugt)."
            ),
            "keep_A": (
                f"Graph von {self.label_B} ist in {self.label_A} enthalten – "
                f"{self.label_A} ist die umfangreichere Struktur."
            ),
            "keep_B": (
                f"Graph von {self.label_A} ist in {self.label_B} enthalten – "
                f"{self.label_B} ist die umfangreichere Struktur."
            ),
            "keep_both": (
                "Keine strukturelle Inklusion feststellbar – "
                "Regelkreise sind strukturell verschieden."
            ),
        }
        return mapping.get(self.decision, f"Unbekannte Entscheidung: {self.decision}")


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

def verify_loops(
    plant_A: TransferFunction,
    controller_A: TransferFunction,
    plant_B: TransferFunction,
    controller_B: TransferFunction,
    label_A: str = "Regelkreis A",
    label_B: str = "Regelkreis B",
    use_adjacency_list: bool = False,
) -> VerificationResult:
    """
    Überprüft zwei Regelkreise auf graphenstrukturelle Äquivalenz.

    Der Subgraph Algorithmus vergleicht die Adjazenzmatrizen der
    Signalflussgraphen beider offener Kreise L_A und L_B.

    Parameters
    ----------
    plant_A, controller_A : Strecke und Regler von Regelkreis A.
    plant_B, controller_B : Strecke und Regler von Regelkreis B.
    label_A, label_B : Beschriftungen für die Anzeige.
    use_adjacency_list : Falls True, wird die Adjazenzlisten-Variante
                         des Algorithmus verwendet (für dichte Graphen).

    Returns
    -------
    VerificationResult mit Entscheidung und Matrizen.

    Raises
    ------
    ImportError : Wenn das ``subgraph``-Paket nicht installiert ist.
    """
    try:
        from subgraph import Subgraph  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Das Paket 'subgraph' ist nicht installiert.\n"
            "Bitte installieren mit:\n"
            "  pip install git+https://gitlab.com/epp-group/subgraph.git@v1.0.0"
        ) from exc

    # Graphen erzeugen
    adj_A, L_A = loop_to_graph(plant_A, controller_A)
    adj_B, L_B = loop_to_graph(plant_B, controller_B)

    # Adjazenzmatrizen auf gemeinsame Größe bringen (padding)
    n_A, n_B = adj_A.shape[0], adj_B.shape[0]
    if n_A != n_B:
        n_max = max(n_A, n_B)
        pad_A = np.zeros((n_max, n_max), dtype=int)
        pad_B = np.zeros((n_max, n_max), dtype=int)
        pad_A[:n_A, :n_A] = adj_A
        pad_B[:n_B, :n_B] = adj_B
        mat_A = pad_A
        mat_B = pad_B
    else:
        mat_A = adj_A.copy()
        mat_B = adj_B.copy()

    # Subgraph Algorithmus ausführen
    algo = Subgraph(use_adjacency_list=use_adjacency_list)
    if use_adjacency_list:
        decision, kept = algo.compare_graphs_with_adj_list(
            mat_A.astype(float), mat_B.astype(float)
        )
    else:
        decision, kept = algo.compare_graphs(
            mat_A.astype(float), mat_B.astype(float)
        )

    return VerificationResult(
        decision=decision,
        matrix_A=adj_A,
        matrix_B=adj_B,
        kept_matrix=kept.astype(int) if kept is not None else None,
        open_loop_A=L_A,
        open_loop_B=L_B,
        label_A=label_A,
        label_B=label_B,
    )

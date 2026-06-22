"""
pylabb.control.bio_classify
============================
Biologisch orientierte Klassifikation von Regelkreisen
in sieben Äquivalenzklassen (Kapitel 5 der Arbeit).

Theoretischer Hintergrund
--------------------------
Regelkreise, die graphenstrukturell äquivalent sind (im Sinne des
Subgraph Algorithmus), können einer gemeinsamen biologischen Regelklasse
zugeordnet werden.  Die Klasse erschließt Entwurfsstrategien und
Erweiterungen, die vom jeweiligen biologischen Vorbild abgeleitet sind.

Sieben biologische Äquivalenzklassen
-------------------------------------
I   – Seestern     (*Asteroidea*)          dezentrale P-Regelung
II  – Pupille      (Pupillenreflex)         PI-Regelung mit Totzeit & Sättigung
III – Kleinhirn    (*Cerebellum*)           PID + Vorsteuerung + ILC
IV  – Herz         (Herzkreislauf)          Kaskadenregelung mit Taktgeber
V   – Albatros     (*Diomedea*)             LQR + Störgrößenaufschaltung
VI  – Mimose       (*Mimosa pudica*)        Nichtlinearer Zweipunkt-Regler (Hysterese)
VII – Tintenfisch  (*Octopus vulgaris*)     Adaptive MIMO-Regelung

Hierarchie (Subgraph-Ordnung, linear)
--------------------------------------
Seestern ⊂ Pupille ⊂ Kleinhirn ⊂ Herz ⊂ Albatros ⊂ Tintenfisch
Mimose ⊃ Seestern, aber Mimose ⊄ Pupille (nichtlinear)

Öffentliche API
---------------
    BioClass           – Enum der sieben Klassen (+ UNCLASSIFIED)
    BioExtension       – Empfohlene Erweiterung (Name, Beschreibung, Code-Hinweis)
    BioClassResult     – Klassifikationsergebnis mit Konfidenz und Subgraph-Kette
    classify_loop      – Hauptfunktion: Regelkreis → BioClassResult
    get_extensions     – BioClass → Liste empfohlener Erweiterungen
    get_reference_topology – BioClass → binäre Referenz-Adjazenzmatrix

Interne Hilfsfunktionen (öffentlich für Tests)
-----------------------------------------------
    _has_integrator
    _has_derivative
    _pad_matrix
    _matrix_contains_reference
    _compute_subgraph_chain
    _classify_by_rules
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from pylabb.core.transfer_function import TransferFunction
from pylabb.control.verification import loop_to_graph


# ===========================================================================
# Enum: Sieben biologische Äquivalenzklassen
# ===========================================================================

class BioClass(enum.Enum):
    """Sieben biologische Äquivalenzklassen plus UNCLASSIFIED."""

    SEESTERN    = (1, "Seestern")
    PUPILLE     = (2, "Pupille")
    KLEINHIRN   = (3, "Kleinhirn")
    HERZ        = (4, "Herz")
    ALBATROS    = (5, "Albatros")
    MIMOSE      = (6, "Mimose")
    TINTENFISCH = (7, "Tintenfisch")
    UNCLASSIFIED = (0, "Unklassifiziert")

    def __init__(self, class_id: int, label: str) -> None:
        self.class_id = class_id
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover
        return f"BioClass.{self.name}(id={self.class_id}, label='{self.label}')"


# ===========================================================================
# Referenztopologien
# ===========================================================================

#: Binäre Adjazenzmatrizen minimaler Dimension n_min × n_min.
#: Jede Matrix kodiert die charakteristischen Kanten des Regelkreistyps.
#:
#: Herleitung:
#:   I   Seestern    n=2: Pfadgraph P_2 mit Rückführkante
#:   II  Pupille     n=3: Integrator-Selbstschleife (M[0,0]=1) + Pfad
#:   III Kleinhirn   n=4: Pupillen-Kern + Vorsteuer-Block (paralleler Pfad)
#:   IV  Herz        n=5: Zwei verschachtelte Schleifen + Taktgeber-Kante
#:   V   Albatros    n=4: Vollständig besetzte letzte Zeile (Zustandsrückkopplung)
#:   VI  Mimose      n=2: Wie Seestern (strukturell); Nichtlinearität separat geprüft
#:   VII Tintenfisch n=4: Annähernd vollständiger Graph (J_4)
_REFERENCE_TOPOLOGIES: Dict[BioClass, NDArray[np.int_]] = {
    BioClass.SEESTERN: np.array(
        [[0, 1],
         [1, 0]],
        dtype=int,
    ),
    BioClass.PUPILLE: np.array(
        [[1, 1, 0],
         [1, 0, 1],
         [0, 0, 0]],
        dtype=int,
    ),
    BioClass.KLEINHIRN: np.array(
        [[1, 1, 0, 0],
         [1, 0, 1, 0],
         [0, 0, 0, 0],
         [1, 1, 0, 1]],
        dtype=int,
    ),
    BioClass.HERZ: np.array(
        [[0, 1, 0, 0, 0],
         [1, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 1],
         [1, 0, 0, 1, 0]],
        dtype=int,
    ),
    BioClass.ALBATROS: np.array(
        [[0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [1, 1, 1, 1]],
        dtype=int,
    ),
    BioClass.MIMOSE: np.array(
        [[0, 1],
         [1, 0]],
        dtype=int,
    ),
    BioClass.TINTENFISCH: np.ones((4, 4), dtype=int),
}

#: Mindestordnung jeder biologischen Klasse (aus Tabelle 5 der Arbeit).
_N_MIN: Dict[BioClass, int] = {
    BioClass.SEESTERN:    2,
    BioClass.PUPILLE:     3,
    BioClass.KLEINHIRN:   4,
    BioClass.HERZ:        5,
    BioClass.ALBATROS:    4,
    BioClass.MIMOSE:      2,
    BioClass.TINTENFISCH: 4,
    BioClass.UNCLASSIFIED: 0,
}

#: Lineare Hierarchie (ohne Mimose, die seitlich steht).
_LINEAR_HIERARCHY: List[BioClass] = [
    BioClass.SEESTERN,
    BioClass.PUPILLE,
    BioClass.KLEINHIRN,
    BioClass.HERZ,
    BioClass.ALBATROS,
    BioClass.TINTENFISCH,
]


# ===========================================================================
# Datenklassen
# ===========================================================================

@dataclass
class BioExtension:
    """Empfohlene Erweiterung für eine biologische Äquivalenzklasse.

    Attributes
    ----------
    name : Kurzname der Erweiterung (z. B. ``"Anti-Windup"``).
    description : Ausführliche Beschreibung der Erweiterung.
    code_hint : Optionaler Python/MicroPython-Hinweis.
    """

    name: str
    description: str
    code_hint: str = ""


@dataclass
class BioClassResult:
    """Ergebnis der biologischen Klassifikation eines Regelkreises.

    Attributes
    ----------
    bio_class : Zugeordnete biologische Äquivalenzklasse.
    confidence : Konfidenz der Zuordnung (0.0 … 1.0).
    subgraph_chain : Vollständige Subgraph-Kette unterhalb der erkannten Klasse.
    adjacency_matrix : Adjazenzmatrix des offenen Kreises M_L.
    density : Kantendichte = nnz(M_L) / n².
    has_self_loop : True, wenn mindestens ein Diagonaleintrag von M_L gleich 1.
    has_full_last_row : True, wenn die letzte Zeile von M_L ausschließlich Einsen enthält.
    controller_has_integrator : True, wenn C(s) einen Pol bei s = 0 hat.
    controller_has_derivative : True, wenn C(s) unecht gebrochen ist (D-Anteil).
    """

    bio_class: BioClass
    confidence: float
    subgraph_chain: List[BioClass]
    adjacency_matrix: NDArray[np.int_]
    density: float
    has_self_loop: bool
    has_full_last_row: bool
    controller_has_integrator: bool
    controller_has_derivative: bool

    @property
    def is_classified(self) -> bool:
        """True, wenn der Regelkreis einer Klasse zugeordnet werden konnte."""
        return self.bio_class is not BioClass.UNCLASSIFIED

    @property
    def summary(self) -> str:
        """Lesbare Zusammenfassung des Klassifikationsergebnisses."""
        if self.bio_class is BioClass.UNCLASSIFIED:
            return (
                "Regelkreis konnte keiner biologischen Äquivalenzklasse "
                "zugeordnet werden."
            )
        chain_str = " \u2286 ".join(c.label for c in self.subgraph_chain)
        return (
            f"Biologische Klasse: {self.bio_class.class_id} "
            f"({self.bio_class.label})\n"
            f"Konfidenz: {self.confidence:.0%}\n"
            f"Subgraph-Kette: {chain_str}"
        )


# ===========================================================================
# Interne Hilfsfunktionen
# ===========================================================================

def _has_integrator(tf: TransferFunction, threshold: float = 1e-6) -> bool:
    """Prüft, ob die Übertragungsfunktion einen Pol bei s = 0 enthält (Integralanteil).

    Parameters
    ----------
    tf : Zu prüfende Übertragungsfunktion.
    threshold : Schwellwert für „nahe Null" im Realteil des konstanten Nennerglieds.

    Returns
    -------
    True, wenn der Konstantterm des Nenners (realer Teil) kleiner als ``threshold``.
    """
    den_real = np.real(tf.den)
    return bool(abs(den_real[-1]) < threshold)


def _has_derivative(tf: TransferFunction) -> bool:
    """Prüft, ob die Übertragungsfunktion einen Differenzialanteil besitzt.

    Kriterium: Der Zählergrad ist **echt größer** als der Nennergrad
    (unecht gebrochene Übertragungsfunktion, z. B. PID mit ``num=[Kd,Kp,Ki]``,
    ``den=[1,0]``).

    Parameters
    ----------
    tf : Zu prüfende Übertragungsfunktion.

    Returns
    -------
    True, wenn ``len(num) > len(den)``.
    """
    return len(tf.num) > len(tf.den)


def _pad_matrix(M: NDArray, target_n: int) -> NDArray:
    """Erweitert eine quadratische Matrix durch Null-Padded auf ``target_n × target_n``.

    Parameter
    ---------
    M : Quadratische Eingangsmatrix (n × n).
    target_n : Zielgröße.  Falls ``n >= target_n``, wird M unverändert zurückgegeben.

    Returns
    -------
    Quadratische Matrix der Größe ``max(n, target_n) × max(n, target_n)``.
    """
    n = M.shape[0]
    if n >= target_n:
        return M.copy()
    padded = np.zeros((target_n, target_n), dtype=M.dtype)
    padded[:n, :n] = M
    return padded


def _matrix_contains_reference(
    M: NDArray[np.int_],
    R: NDArray[np.int_],
) -> float:
    """Berechnet den Anteil der Referenzkanten, der in M enthalten ist.

    Beide Matrizen werden auf die Größe ``max(n_M, n_R) × max(n_M, n_R)``
    gepaddet; dann gilt: Konfidenz = |{(i,j) : R[i,j]=1 ∧ M[i,j]=1}| / |{(i,j) : R[i,j]=1}|.

    Parameters
    ----------
    M : Adjazenzmatrix des zu klassifizierenden Regelkreises.
    R : Referenz-Adjazenzmatrix der biologischen Klasse.

    Returns
    -------
    Wert in [0.0, 1.0].  0.0, wenn R keine Kanten hat oder M zu klein ist.
    """
    n_R = R.shape[0]
    n_M = M.shape[0]
    ref_edges = int(R.sum())
    if ref_edges == 0:
        return 1.0

    n_target = max(n_M, n_R)
    M_padded = _pad_matrix(M.astype(int), n_target)
    R_padded = _pad_matrix(R.astype(int), n_target)

    found = int(np.sum(R_padded & M_padded))
    return found / ref_edges


def _compute_subgraph_chain(bio_class: BioClass) -> List[BioClass]:
    """Berechnet die Subgraph-Kette unterhalb der erkannten Klasse.

    Für die lineare Hierarchie (Seestern … Tintenfisch) enthält die Kette
    alle Klassen bis einschließlich ``bio_class``.
    Die Mimose steht seitlich: Kette = [Seestern, Mimose].
    UNCLASSIFIED liefert eine leere Liste.

    Parameters
    ----------
    bio_class : Zugeordnete biologische Klasse.

    Returns
    -------
    Geordnete Liste von BioClass-Werten (aufsteigend nach Komplexität).
    """
    if bio_class is BioClass.UNCLASSIFIED:
        return []
    if bio_class is BioClass.MIMOSE:
        return [BioClass.SEESTERN, BioClass.MIMOSE]
    if bio_class in _LINEAR_HIERARCHY:
        idx = _LINEAR_HIERARCHY.index(bio_class)
        return list(_LINEAR_HIERARCHY[: idx + 1])
    # Fallback: einzelne Klasse
    return [bio_class]  # pragma: no cover


def _classify_by_rules(
    *,
    density: float,
    has_self_loop: bool,
    has_full_last_row: bool,
    c_has_integrator: bool,
    c_has_derivative: bool,
    has_feedforward: bool,
    has_cascade: bool,
    is_full_state_feedback: bool,
    is_mimo: bool,
    has_hysteresis: bool,
    has_disturbance_channel: bool,
) -> BioClass:
    """Regelbasierte Klassifikation nach der Hierarchie aus Abschnitt 5.5 der Arbeit.

    Die Entscheidungsreihenfolge folgt dem Klassifikationsablauf aus der Arbeit:
    beginnend bei der spezifischsten Klasse (Tintenfisch, VII) absteigend bis
    zur allgemeinsten (Seestern, I).

    Parameters
    ----------
    density : Kantendichte der Adjazenzmatrix.
    has_self_loop : Diagonaleintrag ≠ 0 vorhanden.
    has_full_last_row : Letzte Zeile vollständig besetzt.
    c_has_integrator : Regler hat Pol bei s = 0.
    c_has_derivative : Regler hat D-Anteil (unecht gebrochen).
    has_feedforward : Vorsteuerung vorhanden (Flag vom Aufrufer).
    has_cascade : Kaskadenstruktur vorhanden (Flag).
    is_full_state_feedback : LQR-Zustandsrückkopplung (Flag).
    is_mimo : Mehrgrößensystem (Flag).
    has_hysteresis : Hystereseverhalten vorhanden (Flag oder ``hysteresis_band > 0``).
    has_disturbance_channel : Messbarer Störgrößenkanal vorhanden (Flag).

    Returns
    -------
    BioClass-Wert entsprechend der Klassifikationsregeln.
    """
    # VII – Tintenfisch: MIMO oder sehr dichte Adjazenzmatrix
    if is_mimo or density > 0.75:
        return BioClass.TINTENFISCH

    # V – Albatros: Vollständige Zustandsrückkopplung (LQR)
    #   Erkennungsmerkmale: LQR-Flag ODER vollständig besetzte letzte Zeile
    #   kombiniert mit Störkanal.
    if is_full_state_feedback or (has_full_last_row and has_disturbance_channel):
        return BioClass.ALBATROS

    # IV – Herz: Kaskadenregelung
    if has_cascade:
        return BioClass.HERZ

    # III – Kleinhirn: Vorsteuerung zusätzlich zum Rückkopplungsregler
    if has_feedforward:
        return BioClass.KLEINHIRN

    # VI – Mimose: Nichtlinearer Zweipunktregler mit Hysterese
    #   (Parallelprüfung zur linearen Hierarchie wie in der Arbeit beschrieben)
    if has_hysteresis:
        return BioClass.MIMOSE

    # II – Pupille: Integralanteil im Regler (PI, I oder PID ohne FF)
    if c_has_integrator:
        return BioClass.PUPILLE

    # I – Seestern: Rein proportionaler Regler (kein I- oder D-Anteil erzwingt
    #   eine höhere Klasse, und keines der obigen Flags ist gesetzt)
    return BioClass.SEESTERN


# ===========================================================================
# Klassifikations-Erweiterungen
# ===========================================================================

#: Pro Klasse empfohlene Erweiterungen (aus der Arbeit, Abschnitt 5.x.x).
_EXTENSIONS: Dict[BioClass, List[BioExtension]] = {
    BioClass.SEESTERN: [
        BioExtension(
            name="Konsens-Mehrfachregler",
            description=(
                "Hinzufügen von Kommunikationspfaden zwischen bisher isolierten "
                "P-Reglern zu einem konsens-basierten Mehrfachregler ohne zentrale "
                "Instanz (Ant-Colony-Analogie)."
            ),
            code_hint="consensus_p_controller(nodes, coupling_gain)",
        ),
        BioExtension(
            name="Redundante Auslegung",
            description=(
                "P-Regler in sicherheitskritischen Systemen redundant auslegen: "
                "Bei Ausfall eines Knotens übernehmen die verbleibenden."
            ),
            code_hint="",
        ),
        BioExtension(
            name="Iterative Kp-Erhöhung",
            description=(
                "Kp iterativ von klein beginnend erhöhen (wie der Seestern "
                "experimentierend die Bewegungsrichtung sucht), bis die "
                "gewünschte Dynamik erreicht ist."
            ),
            code_hint="",
        ),
    ],
    BioClass.PUPILLE: [
        BioExtension(
            name="Anti-Windup",
            description=(
                "Anti-Windup-Mechanismus implementieren, sobald eine "
                "Sättigungsgrenze erkennbar ist (analog zur neuronalen "
                "Anpassung beim Pupillenreflex)."
            ),
            code_hint="anti_windup_pi(kp, ti, u_max)",
        ),
        BioExtension(
            name="Smith-Prädiktor",
            description=(
                "Totzeitkompensation nach Smith: Totzeit aus der "
                "Rückkopplungsschleife herausnehmen, um Hippus-ähnliche "
                "Oszillationen zu vermeiden."
            ),
            code_hint="smith_predictor(plant, deadtime)",
        ),
        BioExtension(
            name="Logarithmische Skalierung",
            description=(
                "Für Regelstrecken mit > 40 dB Dynamikbereich logarithmische "
                "Vorverarbeitung der Eingangsgröße (Weber-Fechner-Gesetz). "
                "Reglerverstärkung K_P(y) = K_P0 / (1 + α·ln(1 + y/y0))."
            ),
            code_hint="gain_scheduled_kp(kp0, alpha, y0)",
        ),
    ],
    BioClass.KLEINHIRN: [
        BioExtension(
            name="Iterative Learning Control (ILC)",
            description=(
                "Für repetitive Prozesse: u_{k+1}(t) = u_k(t) + γ·e_k(t). "
                "Jeder Zyklus verbessert den Vorsteuerterm (pylabb.control.ilc)."
            ),
            code_hint="ilc_update(u_prev, error_prev, gamma=0.5)",
        ),
        BioExtension(
            name="Modellbasierte Vorsteuerung",
            description=(
                "Optimales Vorsteuerglied F(s) = Ĝ⁻¹(s) (Streckeninverse, "
                "kausaler properer Anteil). PyLab prüft Kleinhirn-Subgraph."
            ),
            code_hint="feedforward_inverse(plant_model)",
        ),
        BioExtension(
            name="Störgrößenbeobachter",
            description=(
                "Disturbance Observer dem PID-Regler überlagern "
                "(analog zum Kleinhirn, das Trägheitskräfte schätzt)."
            ),
            code_hint="disturbance_observer(plant_model, bandwidth)",
        ),
    ],
    BioClass.HERZ: [
        BioExtension(
            name="Kaskadenregelung",
            description=(
                "Innerer Regelkreis zuerst einstellen (Ziegler-Nichols/IMC), "
                "dann äußeren Regler parametrieren. "
                "Faustregel: ω_innen ≥ 5·ω_außen."
            ),
            code_hint="cascade_control(inner_ctrl, outer_ctrl)",
        ),
        BioExtension(
            name="Bandbreiten-Verhältnis-Überwachung",
            description=(
                "Warnung, wenn ω_innen < 3·ω_außen "
                "(drohende Instabilität des Kaskadensystems, analog EKG)."
            ),
            code_hint="check_bandwidth_ratio(omega_inner, omega_outer, limit=3.0)",
        ),
        BioExtension(
            name="PWM-Takt-Template",
            description=(
                "genmicropy generiert einen biologisch motivierten "
                "'Herz-Template': innerer Kreis als PWM-Regler mit variabler "
                "Frequenz, äußerer PI mit Faktor-10-kleinerer Abtastrate."
            ),
            code_hint="genmicropy_herz_template(inner_ctrl, outer_ctrl, ts_inner, ts_outer)",
        ),
    ],
    BioClass.ALBATROS: [
        BioExtension(
            name="Störgrößenaufschaltung",
            description=(
                "Messbare Störgrößen direkt in den Regler aufnehmen "
                "(Albatros misst Wind explizit). Reduziert erforderliche "
                "Regelverstärkung erheblich."
            ),
            code_hint="disturbance_feedforward(disturbance_gain)",
        ),
        BioExtension(
            name="Riccati-LQR-Synthese",
            description=(
                "LQR-Verstärkungsvektor k über die Riccati-Gleichung "
                "berechnen und in PyLab anzeigen."
            ),
            code_hint="lqr_design(A, B, Q, R_matrix)",
        ),
        BioExtension(
            name="MPC-Erweiterung",
            description=(
                "Model Predictive Control: Optimierung über Horizont H unter "
                "Ausnutzung gemessener Windvorhersage / Störgrößen-Prädiktion."
            ),
            code_hint="mpc_design(plant, horizon=10, constraints=None)",
        ),
    ],
    BioClass.MIMOSE: [
        BioExtension(
            name="Optimale Hysteresebandbreite",
            description=(
                "Hysterese h ≈ 2·σ_n wählen "
                "(σ_n = Standardabweichung des Messrauschens), um echte "
                "Bedrohungen von Windböen/Rauschen zu trennen."
            ),
            code_hint="optimal_hysteresis(noise_std)",
        ),
        BioExtension(
            name="Grenzzyklus-Berechnung",
            description=(
                "Amplitude und Frequenz des Grenzzyklus über die Describing "
                "Function automatisch berechnen und anzeigen."
            ),
            code_hint="limit_cycle_analysis(plant, u_max, hysteresis_band)",
        ),
        BioExtension(
            name="Mindest-Schaltdauer (Dwell Time)",
            description=(
                "Schaltfrequenzbegrenzung als Schutzmechanismus gegen "
                "Stellgliedverschleiß (analog zur Habituierung der Mimose)."
            ),
            code_hint="dwell_time_controller(u_max, min_dwell_ms=50)",
        ),
    ],
    BioClass.TINTENFISCH: [
        BioExtension(
            name="Entkopplungsregler",
            description=(
                "Entkopplungsregler entwirft, der Quereinflüsse zwischen "
                "MIMO-Kanälen kompensiert. Signal: 'Entkopplung erforderlich!'."
            ),
            code_hint="decoupling_controller(plant_matrix)",
        ),
        BioExtension(
            name="Impedanzregelung",
            description=(
                "Adaptive Steifigkeit der Tintenfischarme als Vorbild für "
                "impedanzgeregelte Manipulatoren: "
                "Z_ref(s) = M·s² + D·s + K als Referenzverhalten."
            ),
            code_hint="impedance_control(M, D, K)",
        ),
        BioExtension(
            name="MIMO-MPC",
            description=(
                "Modellprädiktive Mehrgrößenregelung: Koordinations-"
                "mechanismus nach Tintenfisch-Vorbild (optimale Eingaben "
                "für alle Kanäle gleichzeitig)."
            ),
            code_hint="mimo_mpc_design(plant_matrix, horizon, Q, R_matrix)",
        ),
    ],
    BioClass.UNCLASSIFIED: [],
}


# ===========================================================================
# Öffentliche API
# ===========================================================================

def get_reference_topology(bio_class: BioClass) -> NDArray[np.int_]:
    """Gibt die binäre Referenz-Adjazenzmatrix für eine biologische Klasse zurück.

    Parameters
    ----------
    bio_class : Biologische Äquivalenzklasse (darf nicht UNCLASSIFIED sein).

    Returns
    -------
    Kopie der n_min × n_min Referenzmatrix.

    Raises
    ------
    KeyError : Wenn ``bio_class`` keine zugeordnete Referenztopologie hat
               (nur für UNCLASSIFIED).
    """
    return _REFERENCE_TOPOLOGIES[bio_class].copy()


def get_extensions(bio_class: BioClass) -> List[BioExtension]:
    """Gibt die empfohlenen Erweiterungen für eine biologische Klasse zurück.

    Parameters
    ----------
    bio_class : Biologische Äquivalenzklasse.

    Returns
    -------
    Liste von BioExtension-Objekten.  Leere Liste für UNCLASSIFIED.
    """
    return list(_EXTENSIONS.get(bio_class, []))


def classify_loop(
    plant: TransferFunction,
    controller: TransferFunction,
    *,
    has_feedforward: bool = False,
    has_cascade: bool = False,
    is_full_state_feedback: bool = False,
    is_mimo: bool = False,
    has_hysteresis: bool = False,
    hysteresis_band: float = 0.0,
    has_disturbance_channel: bool = False,
    threshold: float = 1e-9,
) -> BioClassResult:
    """Klassifiziert einen Regelkreis in eine biologische Äquivalenzklasse.

    Der Algorithmus folgt dem Klassifikationsablauf aus Abschnitt 5.5 der Arbeit:

    1. Adjazenzmatrix M_L des offenen Kreises L(s) = C(s)·G(s) berechnen.
    2. Strukturmerkmale der Matrix extrahieren (Dichte, Selbstschleifen usw.).
    3. Reglerstruktur analysieren (Integralanteil, Differenzialanteil).
    4. Regelbasierte Klassifikation von Tintenfisch (VII) absteigend zu Seestern (I).
    5. Konfidenz als Anteil übereinstimmender Kanten mit der Referenztopologie berechnen.
    6. Subgraph-Kette zusammenstellen.

    Parameters
    ----------
    plant : Strecken-Übertragungsfunktion G(s).
    controller : Regler-Übertragungsfunktion C(s).
    has_feedforward : True, wenn eine additive Vorsteuerung vorhanden ist.
    has_cascade : True, wenn eine Kaskadenstruktur vorliegt.
    is_full_state_feedback : True, wenn ein Zustandsrückführungsregler (LQR) verwendet wird.
    is_mimo : True, wenn das System mehr als eine Eingangs-/Ausgangsgröße hat.
    has_hysteresis : True, wenn der Regler Hystereseverhalten zeigt.
    hysteresis_band : Hysteresebandbreite h (> 0 impliziert ``has_hysteresis=True``).
    has_disturbance_channel : True, wenn eine messbare Störgröße direkt aufgeschaltet wird.
    threshold : Schwellwert für Adjazenzeinträge (Standard: 1e-9).

    Returns
    -------
    BioClassResult mit Klassenzuweisung, Konfidenz und Subgraph-Kette.

    Examples
    --------
    >>> from pylabb.core.transfer_function import TransferFunction
    >>> G = TransferFunction([1.0], [1.0, 1.0])          # PT1
    >>> C = TransferFunction([2.0], [1.0])                # P-Regler
    >>> result = classify_loop(G, C)
    >>> result.bio_class
    BioClass.SEESTERN
    >>> result.confidence > 0
    True

    >>> G2 = TransferFunction([1.0], [1.0, 1.4, 1.0])    # PT2
    >>> C2 = TransferFunction([1.0, 1.0], [1.0, 0.0])    # PI-Regler
    >>> result2 = classify_loop(G2, C2)
    >>> result2.bio_class
    BioClass.PUPILLE
    """
    # -----------------------------------------------------------------------
    # Schritt 1: Adjazenzmatrix erzeugen
    # -----------------------------------------------------------------------
    adj, _ = loop_to_graph(plant, controller, threshold=threshold)
    n = adj.shape[0]

    # -----------------------------------------------------------------------
    # Schritt 2: Strukturmerkmale extrahieren
    # -----------------------------------------------------------------------
    density: float = float(adj.sum()) / (n * n) if n > 0 else 0.0
    has_self_loop = bool(np.any(np.diag(adj) != 0))
    has_full_last_row = n > 0 and bool(np.all(adj[-1, :] != 0))

    # -----------------------------------------------------------------------
    # Schritt 3: Reglerstruktur analysieren
    # -----------------------------------------------------------------------
    c_has_integrator = _has_integrator(controller, threshold=1e-6)
    c_has_derivative = _has_derivative(controller)

    # Hysterese-Flag zusammenführen
    hysteresis_active = has_hysteresis or (hysteresis_band > 0.0)

    # -----------------------------------------------------------------------
    # Schritt 4: Regelbasierte Klassifikation
    # -----------------------------------------------------------------------
    bio_class = _classify_by_rules(
        density=density,
        has_self_loop=has_self_loop,
        has_full_last_row=has_full_last_row,
        c_has_integrator=c_has_integrator,
        c_has_derivative=c_has_derivative,
        has_feedforward=has_feedforward,
        has_cascade=has_cascade,
        is_full_state_feedback=is_full_state_feedback,
        is_mimo=is_mimo,
        has_hysteresis=hysteresis_active,
        has_disturbance_channel=has_disturbance_channel,
    )

    # -----------------------------------------------------------------------
    # Schritt 5: Konfidenz berechnen
    # -----------------------------------------------------------------------
    if bio_class is BioClass.UNCLASSIFIED:
        confidence = 0.0
    else:
        ref = _REFERENCE_TOPOLOGIES[bio_class]
        raw_conf = _matrix_contains_reference(adj, ref)
        # Kleiner Bonus für strukturell passendes Ergebnis (die Regelregeln
        # liefern eine stärkere Garantie als die reine Adjazenz-Überlappung).
        confidence = min(1.0, raw_conf + 0.05)

    # -----------------------------------------------------------------------
    # Schritt 6: Subgraph-Kette
    # -----------------------------------------------------------------------
    subgraph_chain = _compute_subgraph_chain(bio_class)

    return BioClassResult(
        bio_class=bio_class,
        confidence=confidence,
        subgraph_chain=subgraph_chain,
        adjacency_matrix=adj,
        density=density,
        has_self_loop=has_self_loop,
        has_full_last_row=has_full_last_row,
        controller_has_integrator=c_has_integrator,
        controller_has_derivative=c_has_derivative,
    )

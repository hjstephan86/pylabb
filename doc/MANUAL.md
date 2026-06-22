# PyLab – Ausführliches Anwenderhandbuch

**Version 2.1.0 | Autor: Stephan Epp**

---

## Inhaltsverzeichnis

1. [Überblick](#1-überblick)
2. [Installation](#2-installation)
3. [Schnellstart](#3-schnellstart)
4. [Modulübersicht](#4-modulübersicht)
5. [Modul `core` – Grundlegende Datentypen](#5-modul-core--grundlegende-datentypen)
   - 5.1 [Übertragungsfunktion (`TransferFunction`)](#51-übertragungsfunktion-transferfunction)
   - 5.2 [Zustandsraumdarstellung (`StateSpace`)](#52-zustandsraumdarstellung-statespace)
   - 5.3 [Signale (`Signal`)](#53-signale-signal)
   - 5.4 [Mathematische Hilfsfunktionen (`math_utils`)](#54-mathematische-hilfsfunktionen-math_utils)
6. [Modul `control` – Reglerentwurf](#6-modul-control--reglerentwurf)
   - 6.1 [PID-Regler (`pid`)](#61-pid-regler-pid)
   - 6.2 [Reglerentwurf (`design`)](#62-reglerentwurf-design)
   - 6.3 [Stabilitätsanalyse (`stability`)](#63-stabilitätsanalyse-stability)
   - 6.4 [Regelkreis-Äquivalenzprüfung (`verification`)](#64-regelkreis-äquivalenzprüfung-verification)
   - 6.5 [Biologische Äquivalenzklassen (`bio_classify`)](#65-biologische-äquivalenzklassen-bio_classify)
7. [Modul `simulation` – Zeitbereichssimulation](#7-modul-simulation--zeitbereichssimulation)
8. [Modul `visualization` – Grafische Darstellung](#8-modul-visualization--grafische-darstellung)
9. [Modul `codegen` – MicroPython-Codegenerierung](#9-modul-codegen--micropython-codegenerierung)
10. [GUI – Grafische Benutzeroberfläche](#10-gui--grafische-benutzeroberfläche)
11. [Vollständige Beispiele](#11-vollständige-beispiele)
12. [Tests und Code Coverage](#12-tests-und-code-coverage)
13. [Projektstruktur](#13-projektstruktur)

---

## 1. Überblick

**PyLab** ist ein umfassendes Python-Framework für:

- **Regelungstechnik**: Übertragungsfunktionen, Zustandsraum, PID-Entwurf, Stabilitätsanalyse
- **Simulation**: Zeitbereichssimulation von Regelkreisen mit beliebigen Eingangssignalen
- **Visualisierung**: Bode-Diagramm, Nyquist-Ortskurve, Pol-Nullstellen-Plan, Sprungantworten
- **MicroPython-Codegenerierung**: Automatische Erzeugung von lauffähigem Eingebettet-Code für ESP32, RP2040, STM32
- **Grafische Oberfläche**: Interaktives PyQt6-GUI mit System-Editor, Analyse-Panel und Code-Generator

PyLab eignet sich sowohl für die **skriptbasierte Nutzung in Python** (z. B. in Jupyter-Notebooks oder eigenen Programmen) als auch für die **interaktive Nutzung** über die integrierte GUI.

---

## 2. Installation

### Voraussetzungen

| Paket     | Mindestversion |
|-----------|---------------|
| Python    | ≥ 3.10        |
| numpy     | ≥ 1.24        |
| scipy     | ≥ 1.10        |
| matplotlib| ≥ 3.7         |
| PyQt6     | ≥ 6.4         |

### Installation aus dem Quellcode

```bash
# Repository klonen (oder Quellcode entpacken)
cd /pfad/zum/pylabb

# Aktivieren der virtuellen Environment
python -m venv venv
source venv/bin/activate       # Linux / macOS

# Installation
pip install .

# Optional: Entwicklungsabhängigkeiten (Tests, Linting)
pip install ".[dev]"
```

### GUI starten

```bash
pylabb-gui
```

---

## 3. Schnellstart

```python
from pylabb.core.transfer_function import TransferFunction
from pylabb.control.pid import PIDController, DiscretePIDController
from pylabb.simulation.time_domain import ClosedLoopSimulator
from pylabb.core.signals import step_signal

# 1. Strecke definieren: G(s) = 1 / (s² + 2s + 1)
G = TransferFunction([1], [1, 2, 1])

# 2. PID-Regler
pid = PIDController(Kp=2.0, Ti=1.5, Td=0.2)

# 3. Simulation (Sprung von 0→1 bei t=0)
t, y = G.step_response(t_end=10.0)

# 4. Stabilitätsanalyse
from pylabb.control.stability import analyze_stability
info = analyze_stability(G)
print(info)
```

---

## 4. Modulübersicht

```
pylabb/
├── core/
│   ├── transfer_function.py   – G(s) = B(s)/A(s), Laplace / z-Domain
│   ├── state_space.py         – ẋ = Ax + Bu,  y = Cx + Du
│   ├── signals.py             – Signalgenerierung (Sprung, Sinus, …)
│   └── math_utils.py          – Polynom-, Partialbruch-, ZOH-Hilfsfunktionen
├── control/
│   ├── pid.py                 – PID- und diskrete PID-Regler + Einstellregeln
│   ├── design.py              – Lead/Lag, Notch, LQR, Polvorgabe
│   ├── stability.py           – Routh, Nyquist, Phasen-/Amplitudenrand
│   ├── verification.py        – Graphenstruktur-Äquivalenzprüfung (Subgraph)
│   └── bio_classify.py        – Biologische Äquivalenzklassen (neu in 2.1.0)
├── simulation/
│   └── time_domain.py         – ClosedLoopSimulator, simulate_tf, simulate_ss
├── visualization/
│   ├── bode.py                – Bode-Diagramm
│   ├── nyquist.py             – Nyquist-Ortskurve
│   ├── rlocus.py              – Wurzelortskurve
│   └── time_plots.py          – Zeitverläufe
├── codegen/
│   └── micropython.py         – MicroPython-Codegenerator
└── gui/
    ├── main_window.py         – PyQt6-Hauptfenster
    └── widgets/
        ├── system_editor.py      – System-Editor-Dock
        ├── analysis_widget.py    – Analyse-Panel
        ├── plot_widget.py        – Diagramm-Tabs
        ├── codegen_widget.py     – Codegen-Panel
        └── verification_widget.py – Regelkreis-Äquivalenzprüfungs-Dock
```

---

## 5. Modul `core` – Grundlegende Datentypen

### 5.1 Übertragungsfunktion (`TransferFunction`)

Die Klasse `TransferFunction` repräsentiert eine rationale Laplace- (bzw. z-Bereichs-) Übertragungsfunktion:

$$G(s) = \frac{B(s)}{A(s)} = \frac{b_n s^n + \cdots + b_0}{a_m s^m + \cdots + a_0}$$

#### Konstruktion

```python
from pylabb.core.transfer_function import TransferFunction

# G(s) = 1 / (s² + 2s + 1)
G = TransferFunction([1], [1, 2, 1])

# Diskrete ÜTF mit Abtastzeit dt = 0.01 s
Gd = TransferFunction([0.5, 0.1], [1, -0.8], dt=0.01)

# Mit eigenem Namen
G = TransferFunction([1], [1, 2, 1], name="Strecke")
```

#### Wichtige Methoden

| Methode | Beschreibung |
|--------|-------------|
| `G.poles()` | Pole der ÜTF als NumPy-Array |
| `G.zeros()` | Nullstellen der ÜTF |
| `G.freqresp(omega)` | Frequenzgang $G(j\omega)$, gibt `(omega, H)` zurück |
| `G.step_response(t_end, n_points)` | Sprungantwort, gibt `(t, y)` zurück |
| `G.impulse_response(t_end)` | Impulsantwort |
| `G.to_state_space()` | Konvertierung in Zustandsraumdarstellung |
| `G.discretize(dt, method)` | Diskretisierung: `'zoh'` oder `'tustin'` |
| `G.bode(omega)` | Gibt `(omega, mag_dB, phase_deg)` zurück |

#### Algebraische Operationen

```python
G1 = TransferFunction([1], [1, 1])    # 1/(s+1)
G2 = TransferFunction([2], [1, 2])    # 2/(s+2)

G_serie     = G1 * G2     # Reihenschaltung
G_parallel  = G1 + G2     # Parallelschaltung
G_inv       = 1 / G1      # Invertierung
G_geschl    = G1 / (1 + G1 * G2)  # Geschlossener Kreis (manuell)
```

#### Beispiel: Pol-Nullstellen-Analyse

```python
G = TransferFunction([1, 2], [1, 3, 2])  # (s+2)/((s+1)(s+2))

print("Pole:", G.poles())          # [-2. -1.]
print("Nullstellen:", G.zeros())   # [-2.]

# Sprungantwort plotten
import matplotlib.pyplot as plt
t, y = G.step_response(t_end=10)
plt.plot(t, y)
plt.xlabel("Zeit [s]")
plt.ylabel("Ausgang")
plt.title("Sprungantwort")
plt.grid(True)
plt.show()
```

---

### 5.2 Zustandsraumdarstellung (`StateSpace`)

Beschreibt ein LTI-System in der Form:

$$\dot{x} = A x + B u, \quad y = C x + D u$$

#### Konstruktion

```python
import numpy as np
from pylabb.core.state_space import StateSpace

A = [[-2, 0], [1, -1]]
B = [[1], [0]]
C = [[0, 1]]
D = [[0]]

sys = StateSpace(A, B, C, D)

# Zeitdiskret
sys_d = StateSpace(A, B, C, D, dt=0.01)
```

#### Wichtige Methoden

| Methode | Beschreibung |
|--------|-------------|
| `sys.order` | Systemordnung n |
| `sys.eigenvalues()` | Eigenwerte der Matrix A |
| `sys.is_controllable()` | Prüft Steuerbarkeit (Rang der Steuerbarkeitsmatrix) |
| `sys.is_observable()` | Prüft Beobachtbarkeit |
| `sys.to_transfer_function()` | Konvertierung in ÜTF |
| `sys.step_response(t_end)` | Sprungantwort |
| `sys.simulate(t, u)` | Simulation mit beliebigem Eingangssignal |
| `sys.place_poles(desired_poles)` | Polvorgabe – berechnet Rückführvektor K |
| `sys.lqr(Q, R)` | LQR-Reglerentwurf |
| `sys.discretize(dt, method)` | Diskretisierung per `'zoh'` oder `'euler'` |

#### Beispiel: Polvorgabe

```python
import numpy as np
from pylabb.core.state_space import StateSpace

A = [[0, 1], [-2, -3]]
B = [[0], [1]]
C = [[1, 0]]
D = [[0]]

sys = StateSpace(A, B, C, D)

# Gewünschte Pole bei s = -5 ± 5j
desired_poles = [-5 + 5j, -5 - 5j]
K = sys.place_poles(desired_poles)
print("Rückführvektor K:", K)
```

#### Beispiel: LQR-Entwurf

```python
Q = np.diag([10, 1])  # Zustandsgewichtung
R = np.array([[1]])   # Stellgrößengewichtung

K, S, eig = sys.lqr(Q, R)
print("LQR-Gain K:", K)
print("Geschlossene Kreiseigenwerte:", eig)
```

---

### 5.3 Signale (`Signal`)

Die `Signal`-Klasse kapselt diskrete Signale mit Zeitachse und Metadaten.

#### Fabrikfunktionen

```python
from pylabb.core.signals import (
    step_signal, ramp_signal, sine_signal,
    impulse_signal, square_signal, chirp_signal, noise_signal
)

t_end = 10.0
dt    = 0.01

# Einheitssprung bei t=1s
s_step    = step_signal(t_end, dt, step_time=1.0, amplitude=1.0)

# Rampe
s_ramp    = ramp_signal(t_end, dt, slope=0.5)

# Sinus: 2 Hz, Amplitude 1.5
s_sin     = sine_signal(t_end, dt, frequency=2.0, amplitude=1.5)

# Impuls (Dirac-Näherung)
s_impulse = impulse_signal(t_end, dt)

# Rechtecksignal (50 % Tastverhältnis, 1 Hz)
s_square  = square_signal(t_end, dt, frequency=1.0)

# Chirp (0.1 Hz → 10 Hz)
s_chirp   = chirp_signal(t_end, dt, f0=0.1, f1=10.0)

# Weißes Rauschen (Standardabweichung 0.1)
s_noise   = noise_signal(t_end, dt, std=0.1)
```

#### Signal-Arithmetik

```python
# Überlagerte Signale
s_combined = s_step + s_noise       # Signal + Rauschen
s_scaled   = s_sine * 2.0           # Skalierung

print(s_step.dt)        # Abtastzeit
print(len(s_step.t))    # Anzahl Samples
```

---

### 5.4 Mathematische Hilfsfunktionen (`math_utils`)

```python
from pylabb.core.math_utils import (
    poly_eval, poly_multiply, poly_add,
    roots_of_polynomial, partial_fraction,
    bilinear_transform, zoh_discretize,
    routh_array, count_rhp_roots,
)

# Polynom p(s) = s² + 3s + 2 bei s = 1
val = poly_eval([1, 3, 2], 1.0)   # → 6.0

# Partialbruchzerlegung H(s) = 1/(s²+3s+2) = 1/(s+1) + (-1)/(s+2)
r, p, k = partial_fraction([1], [1, 3, 2])

# Routh-Array für Polynom a(s) = s³ + 2s² + 3s + 4
routh = routh_array([1, 2, 3, 4])
n_unstable = count_rhp_roots([1, 2, 3, 4])
```

---

## 6. Modul `control` – Reglerentwurf

### 6.1 PID-Regler (`pid`)

#### Kontinuierlicher PID-Regler (`PIDController`)

```python
from pylabb.control.pid import PIDController

pid = PIDController(
    Kp = 2.0,    # Proportionalverstärkung
    Ti = 1.5,    # Nachstellzeit [s]
    Td = 0.1,    # Vorhaltzeit [s]
    N  = 20.0,   # D-Filterkoeffizient
)

# Übertragungsfunktion des Reglers C(s)
C = pid.transfer_function()
print(C)
```

Die Übertragungsfunktion lautet:

$$C(s) = K_p \cdot \left(1 + \frac{1}{T_i s} + \frac{T_d s}{1 + \frac{T_d}{N} s}\right)$$

#### Zeitdiskreter PID (`DiscretePIDController`)

```python
from pylabb.control.pid import DiscretePIDController

dpid = DiscretePIDController(
    Kp    = 2.0,
    Ti    = 1.5,
    Td    = 0.1,
    dt    = 0.01,    # Abtastzeit [s]
    u_min = -10.0,   # Stellgrößenbegrenzung
    u_max =  10.0,
    N     = 20.0,
)

# Einzelner Regelschritt
u = dpid.update(setpoint=1.0, measurement=0.8)

# Zustand zurücksetzen
dpid.reset()
```

#### Automatische Einstellregeln

```python
from pylabb.control.pid import (
    ziegler_nichols_step,
    ziegler_nichols_relay,
    cohen_coon,
    lambda_tuning,
)

# Ziegler-Nichols (Sprungversuch): aus FOPDT-Kennwerten
# K=Streckengewinn, T=Zeitkonstante, L=Totzeit
pid_zn = ziegler_nichols_step(K=1.0, T=2.0, L=0.5, mode="PID")

# Cohen-Coon
pid_cc = cohen_coon(K=1.0, T=2.0, L=0.5, mode="PID")

# Lambda-Einstellung (λ = gewünschte Zeitkonstante des geregelten Systems)
pid_lm = lambda_tuning(K=1.0, T=2.0, L=0.5, lam=1.0)
```

---

### 6.2 Reglerentwurf (`design`)

#### Lead/Lag-Kompensatoren

```python
from pylabb.control.design import lead_compensator, lag_compensator, lead_lag_compensator

# Lead-Kompensator: Phasenanhebung (α > 1)
C_lead = lead_compensator(alpha=10.0, T=0.1, K=1.0)

# Lag-Kompensator: Verstärkungsabsenkung bei hohen Frequenzen (α < 1)
C_lag = lag_compensator(alpha=0.1, T=1.0, K=1.0)

# Kombinierter Lead-Lag
C_ll = lead_lag_compensator(T1=0.1, alpha1=10.0, T2=1.0, alpha2=0.1)
```

#### Notch- und Bandpassfilter

```python
from pylabb.control.design import notch_filter, bandpass_filter

# Kerbfilter bei ω₀ = 10 rad/s
C_notch = notch_filter(omega0=10.0, zeta=0.05, K=1.0)

# Bandpassfilter
C_bp = bandpass_filter(omega0=10.0, bandwidth=2.0)
```

#### Offener und geschlossener Kreis

```python
from pylabb.control.design import open_loop, closed_loop, sensitivity

G = TransferFunction([1], [1, 2, 1])
C = TransferFunction([2, 1], [1])

L   = open_loop(C, G)        # L(s) = C(s)·G(s)
T   = closed_loop(C, G)      # T(s) = L/(1+L)  (Führungs-ÜTF)
S   = sensitivity(C, G)      # S(s) = 1/(1+L)  (Empfindlichkeit)
```

---

### 6.3 Stabilitätsanalyse (`stability`)

```python
from pylabb.control.stability import analyze_stability
from pylabb.core.transfer_function import TransferFunction

G = TransferFunction([1, 2], [1, 3, 2, 0])  # Offener Kreis

info = analyze_stability(G)
print(info)
```

**Ausgabe:**
```
=== Stabilitätsanalyse ===
Stabil:              True
Instabile Pole:      0
Amplitudenrand:      12.34 dB
Phasenrand:          45.67°
Durchtrittsfrequenz: 1.414 rad/s
Phasenumkehrfreq.:   3.162 rad/s
Pole: [-2. -1.  0.]
Nullstellen: [-2.]
Dominante Pole: [-1.+0.j]
```

Das `StabilityInfo`-Objekt enthält folgende Felder:

| Attribut | Typ | Beschreibung |
|----------|-----|-------------|
| `is_stable` | `bool` | Systemstabilität (kein RHP-Pol) |
| `gain_margin_dB` | `float` | Amplitudenrand [dB] |
| `phase_margin_deg` | `float` | Phasenrand [°] |
| `gain_crossover_freq` | `float` | Durchtrittsfrequenz [rad/s] |
| `phase_crossover_freq` | `float` | Phasenumkehrfrequenz [rad/s] |
| `rhp_poles` | `int` | Anzahl rechts-halbebener Pole |
| `poles` | `NDArray` | Alle Pole |
| `zeros` | `NDArray` | Alle Nullstellen |
| `dominant_poles` | `NDArray` | Dominante Pole |

#### Routh-Hurwitz-Kriterium (direkt)

```python
from pylabb.core.math_utils import routh_array, count_rhp_roots

# Charakteristisches Polynom: s³ + 2s² + 3s + 4
coeffs = [1, 2, 3, 4]
print(routh_array(coeffs))         # Routh-Schema
print(count_rhp_roots(coeffs))     # Anzahl instabiler Wurzeln
```

---

### 6.4 Regelkreis-Äquivalenzprüfung (`verification`)

Ab Version 2.0.0 bietet PyLab eine graphenstrukturelle Äquivalenzprüfung zweier
Regelkreise auf Basis des **Subgraph Algorithmus** (Abhängigkeit:
`subgraph @ git+https://gitlab.com/epp-group/subgraph.git@v1.0.0`).

#### Funktionsprinzip

Ein Regelkreis aus Strecke $G(s)$ und Regler $C(s)$ wird in einen
gerichteten **Signalflussgraphen** überführt:

1. Offenen Kreis berechnen: $L(s) = C(s) \cdot G(s)$
2. Zustandsraumdarstellung erzeugen (observable canonical form)
3. Systemmatrix **A** → binäre Adjazenzmatrix
   (Eintrag $a_{ij} = 1$ wenn $|A_{ij}| > \varepsilon$)
4. Subgraph Algorithmus prüft via zyklischer Rotationssuche, ob einer
   der beiden Graphen im anderen enthalten ist.

#### Entscheidungen

| Rückgabewert | Bedeutung |
|---|---|
| `"equal"` | Beide Graphen sind strukturell äquivalent |
| `"equal_keep_A"` | Äquivalent; Graph A hat mehr Kanten |
| `"equal_keep_B"` | Äquivalent; Graph B hat mehr Kanten |
| `"keep_A"` | Graph B ist in Graph A enthalten (A ist komplexer) |
| `"keep_B"` | Graph A ist in Graph B enthalten (B ist komplexer) |
| `"keep_both"` | Keine strukturelle Inklusion – Regelkreise sind verschieden |

#### API

```python
from pylabb.core.transfer_function import TransferFunction
from pylabb.control.verification import verify_loops, loop_to_graph

# Regelkreis A: PT2-Strecke + P-Regler
G_A = TransferFunction([1], [1, 1.4, 1], name="PT2")
C_A = TransferFunction([2], [1], name="P")

# Regelkreis B: PT2-Strecke + PI-Regler
G_B = TransferFunction([1], [1, 1.4, 1], name="PT2")
C_B = TransferFunction([2, 1], [1, 0], name="PI")

# Prüfung
result = verify_loops(
    plant_A=G_A, controller_A=C_A,
    plant_B=G_B, controller_B=C_B,
    label_A="P-Regelkreis",
    label_B="PI-Regelkreis",
)

print(result.decision)       # z.B. "keep_b"
print(result.summary)        # Lesbare Erklärung
print(result.are_equivalent) # False
print(result.b_contains_a)   # True (PI-Kreis enthält P-Kreis)
print(result.matrix_A)       # Adjazenzmatrix Regelkreis A
print(result.matrix_B)       # Adjazenzmatrix Regelkreis B
```

#### Nur den Graphen erzeugen

```python
# Adjazenzmatrix und offenen Kreis L(s) abrufen
adj, L = loop_to_graph(G_A, C_A)
print("Adjazenzmatrix:", adj)
print("Offener Kreis:", L)
```

#### `VerificationResult` – Attribute und Properties

| Attribut / Property | Typ | Beschreibung |
|---|---|---|
| `decision` | `str` | Rohe Entscheidung des Algorithmus |
| `summary` | `str` | Lesbare Zusammenfassung |
| `are_equivalent` | `bool` | Wahr bei allen `equal*`-Entscheidungen |
| `a_contains_b` | `bool` | Wahr bei `keep_A` |
| `b_contains_a` | `bool` | Wahr bei `keep_B` |
| `are_independent` | `bool` | Wahr bei `keep_both` |
| `matrix_A` | `ndarray` | Adjazenzmatrix des Regelkreises A |
| `matrix_B` | `ndarray` | Adjazenzmatrix des Regelkreises B |
| `kept_matrix` | `ndarray\|None` | Bevorzugte Matrix (oder `None`) |
| `open_loop_A` | `TransferFunction` | $L_A(s)$ |
| `open_loop_B` | `TransferFunction` | $L_B(s)$ |

#### Adjazenzlisten-Variante (für dichte Graphen)

```python
result = verify_loops(
    G_A, C_A, G_B, C_B,
    use_adjacency_list=True,  # mengenbasierter Vergleich
)
```

---

### 6.5 Biologische Äquivalenzklassen (`bio_classify`)

> **Neu in Version 2.1.0** – implementiert Kapitel 5 der wissenschaftlichen Arbeit
> *„Biologisch orientierte Äquivalenzklassen von Regelkreisen"*.

Das Modul klassifiziert einen Regelkreis in eine von **sieben biologischen
Äquivalenzklassen** auf Basis seiner graphenstrukturellen Eigenschaften.
Jede Klasse ist nach einem biologischen Vorbild benannt und liefert konkrete
**Entwurfsempfehlungen und Erweiterungsmöglichkeiten**.

#### Sieben biologische Klassen

| Nr | Klasse | Biologisches Vorbild | Reglertyp | Graphdichte |
|----|-----------|-----------------------|-----------|-------------|
| I | Seestern | *Asteroidea* | P | gering |
| II | Pupille | Pupillenreflex | PI + Totzeit | mittel |
| III | Kleinhirn | *Cerebellum* | PID + Vorsteuerung | mittel |
| IV | Herz | Herzkreislauf | Kaskadenregelung | mittel |
| V | Albatros | *Diomedea* | LQR + Störaufschaltung | hoch |
| VI | Mimose | *Mimosa pudica* | Zweipunkt + Hysterese | gering (nichtlin.) |
| VII | Tintenfisch | *Octopus vulgaris* | Adaptive MIMO | sehr hoch |

**Subgraph-Hierarchie (linear):** Seestern ⊂ Pupille ⊂ Kleinhirn ⊂ Herz ⊂ Albatros ⊂ Tintenfisch

#### API – Schnellübersicht

```python
from pylabb.core.transfer_function import TransferFunction
from pylabb.control.bio_classify import (
    BioClass,          # Enum der sieben Klassen + UNCLASSIFIED
    classify_loop,     # Hauptfunktion: Regelkreis → BioClassResult
    get_extensions,    # BioClass → Liste empfohlener Erweiterungen
    get_reference_topology,  # BioClass → binäre Referenz-Adjazenzmatrix
    BioClassResult,    # Ergebnisdatenklasse
    BioExtension,      # Erweiterungsdatenklasse
)
```

#### Klassifikations-Beispiele

**Beispiel 1 – P-Regelkreis → Klasse I (Seestern)**

```python
G = TransferFunction([1.0], [1.0, 1.0])   # PT1-Strecke
C = TransferFunction([2.0], [1.0])        # P-Regler

result = classify_loop(G, C)
print(result.bio_class)        # BioClass.SEESTERN
print(result.confidence)       # z. B. 0.55
print(result.subgraph_chain)   # [BioClass.SEESTERN]
print(result.summary)
```

**Beispiel 2 – PI-Regelkreis → Klasse II (Pupille)**

```python
G = TransferFunction([0.8], [300.0, 40.0, 1.0])   # Heizung PT2
C = TransferFunction([2.0, 0.08], [1.0, 0.0])     # PI-Regler

result = classify_loop(G, C)
print(result.bio_class)        # BioClass.PUPILLE
print(result.subgraph_chain)   # [BioClass.SEESTERN, BioClass.PUPILLE]
```

**Beispiel 3 – PID mit Vorsteuerung → Klasse III (Kleinhirn)**

```python
G = TransferFunction([1.0], [1.0, 0.0])            # Integrierende Strecke
C = TransferFunction([0.5, 2.0, 1.0], [1.0, 0.0])  # PID-Regler

result = classify_loop(G, C, has_feedforward=True)
print(result.bio_class)   # BioClass.KLEINHIRN
```

**Beispiel 4 – LQR → Klasse V (Albatros)**

```python
result = classify_loop(G, C,
    is_full_state_feedback=True,
    has_disturbance_channel=True)
print(result.bio_class)   # BioClass.ALBATROS
```

**Beispiel 5 – Zweipunkt-Regler mit Hysterese → Klasse VI (Mimose)**

```python
result = classify_loop(G, C, hysteresis_band=0.5)
print(result.bio_class)   # BioClass.MIMOSE
```

**Beispiel 6 – MIMO-System → Klasse VII (Tintenfisch)**

```python
result = classify_loop(G, C, is_mimo=True)
print(result.bio_class)   # BioClass.TINTENFISCH
```

#### Empfohlene Erweiterungen abrufen

```python
for ext in get_extensions(result.bio_class):
    print(ext.name)
    print(" ", ext.description)
    if ext.code_hint:
        print(" Code-Hinweis:", ext.code_hint)
```

#### Referenztopologie inspizieren

```python
R = get_reference_topology(BioClass.ALBATROS)
print(R)
# [[0 1 0 0]
#  [0 0 1 0]
#  [0 0 0 1]
#  [1 1 1 1]]
```

#### `classify_loop` – Parameter

| Parameter | Typ | Standard | Bedeutung |
|---|---|---|---|
| `plant` | `TransferFunction` | – | Strecken-ÜTF $G(s)$ |
| `controller` | `TransferFunction` | – | Regler-ÜTF $C(s)$ |
| `has_feedforward` | `bool` | `False` | Additive Vorsteuerung vorhanden |
| `has_cascade` | `bool` | `False` | Kaskadenstruktur vorhanden |
| `is_full_state_feedback` | `bool` | `False` | LQR-Zustandsrückkopplung |
| `is_mimo` | `bool` | `False` | Mehrgrößensystem |
| `has_hysteresis` | `bool` | `False` | Hystereseverhalten |
| `hysteresis_band` | `float` | `0.0` | Hysteresebandbreite $h$ |
| `has_disturbance_channel` | `bool` | `False` | Messbare Störgröße aufgeschaltet |
| `threshold` | `float` | `1e-9` | Schwellwert für Adjazenzeinträge |

#### `BioClassResult` – Attribute und Properties

| Attribut / Property | Typ | Beschreibung |
|---|---|---|
| `bio_class` | `BioClass` | Zugeordnete biologische Klasse |
| `confidence` | `float` | Konfidenz der Zuordnung (0 … 1) |
| `subgraph_chain` | `list[BioClass]` | Vollständige Subgraph-Kette |
| `adjacency_matrix` | `ndarray` | Adjazenzmatrix des offenen Kreises |
| `density` | `float` | Kantendichte = nnz(M) / n² |
| `has_self_loop` | `bool` | Diagonaleintrag ≠ 0 vorhanden |
| `has_full_last_row` | `bool` | Letzte Zeile vollständig besetzt |
| `controller_has_integrator` | `bool` | Regler hat Pol bei $s = 0$ |
| `controller_has_derivative` | `bool` | Regler hat D-Anteil |
| `is_classified` | `bool` (property) | `True` wenn nicht UNCLASSIFIED |
| `summary` | `str` (property) | Lesbare Zusammenfassung |

---

## 7. Modul `simulation` – Zeitbereichssimulation

### `ClosedLoopSimulator`

Simuliert einen vollständigen Regelkreis mit Regler und Strecke.

```python
from pylabb.simulation.time_domain import ClosedLoopSimulator
from pylabb.core.transfer_function import TransferFunction
from pylabb.control.pid import PIDController
from pylabb.core.signals import step_signal, noise_signal

# System und Regler
G   = TransferFunction([1], [1, 2, 1])
pid = PIDController(Kp=2.0, Ti=1.5, Td=0.1)
C   = pid.transfer_function()

# Signale
dt   = 0.01
w    = step_signal(t_end=15.0, dt=dt, amplitude=1.0)   # Sollwert
d    = noise_signal(t_end=15.0, dt=dt, std=0.02)       # Störung

# Simulation
sim  = ClosedLoopSimulator(plant=G, controller=C)
result = sim.simulate(setpoint=w, disturbance=d)

# Ergebnisse
print(f"Anstiegszeit:    {result.rise_time():.3f} s")
print(f"Ausregelzeit:    {result.settling_time():.3f} s")
print(f"Überschwingen:    {result.overshoot():.1f} %")
print(f"Bleibende Abw.:  {result.steady_state_error():.4f}")

# Zeitverläufe plotten
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(result.t, result.setpoint, 'k--', label='Sollwert')
plt.plot(result.t, result.y, label='Ausgang')
plt.plot(result.t, result.u, label='Stellgröße')
plt.legend()
plt.grid(True)
plt.show()
```

### `SimulationResult` – Kennwerte

| Methode | Beschreibung |
|--------|-------------|
| `result.rise_time(low, high)` | Anstiegszeit von 10 % auf 90 % des Endwerts |
| `result.settling_time(band)` | Ausregelzeit (Standard: ±2 % Band) |
| `result.overshoot()` | Überschwingen in Prozent |
| `result.steady_state_error()` | Bleibende Regelabweichung |
| `result.iae()` | Integral Absolute Error |
| `result.ise()` | Integral Squared Error |
| `result.itae()` | Integral Time-Weighted Absolute Error |

### Direkte Simulation einzelner Systeme

```python
from pylabb.simulation.time_domain import simulate_tf, simulate_ss
import numpy as np

G = TransferFunction([1], [1, 2, 1])

# Beliebiges Eingangssignal
t = np.linspace(0, 10, 1000)
u = np.sin(2 * np.pi * t)

t_out, y_out = simulate_tf(G, t, u)
```

---

## 8. Modul `visualization` – Grafische Darstellung

Alle Visualisierungsfunktionen geben `matplotlib`-Figuren zurück und können in eigene Plots eingebettet werden.

### Bode-Diagramm

```python
from pylabb.visualization.bode import bode_plot
from pylabb.core.transfer_function import TransferFunction

G = TransferFunction([1], [1, 2, 1])

# Einfacher Plot
fig = bode_plot(G)

# Mehrere Systeme vergleichen
G2  = TransferFunction([2], [1, 3, 2])
fig = bode_plot([G, G2], labels=["G1", "G2"], omega_range=(0.01, 100))
```

### Nyquist-Ortskurve

```python
from pylabb.visualization.nyquist import nyquist_plot

fig = nyquist_plot(G)
```

### Wurzelortskurve

```python
from pylabb.visualization.rlocus import rlocus_plot

fig = rlocus_plot(G, k_range=(0, 50))
```

### Zeitverläufe

```python
from pylabb.visualization.time_plots import plot_simulation_result

fig = plot_simulation_result(result)       # SimulationResult-Objekt
```

---

## 9. Modul `codegen` – MicroPython-Codegenerierung

Das `codegen`-Modul erzeugt vollständige, eigenständige MicroPython-Quelldateien, die direkt auf Mikrocontroller (ESP32, RP2040, STM32 mit MicroPython ≥ 1.20) übertragen und ausgeführt werden können – ohne externe Abhängigkeiten.

### Konfiguration (`CodegenConfig`)

```python
from pylabb.codegen.micropython import CodegenConfig

cfg = CodegenConfig(
    float_type       = "float",          # 'float' oder 'double'
    indent           = "    ",           # Einrückung
    add_comments     = True,             # Docstrings/Kommentare einfügen
    target           = "ESP32 / RP2040", # Zielplattform-Hinweis im Header
    tick_source      = "time.ticks_ms",  # Zeitquelle: 'time.ticks_ms' |
                                         #   'machine.RTC' | 'custom'
    include_uart     = False,            # UART-Ausgabe einbinden
    include_watchdog = False,            # Watchdog-Reset einbinden
)
```

### PID-Regler generieren (`gen_pid`)

```python
from pylabb.codegen.micropython import gen_pid, CodegenConfig
from pylabb.control.pid import PIDController, DiscretePIDController

# Aus kontinuierlichem Regler
pid = PIDController(Kp=1.5, Ti=2.0, Td=0.1)
code = gen_pid(pid, class_name="MyPID")

# Aus zeitdiskretem Regler (empfohlen – dt ist damit eindeutig)
dpid = DiscretePIDController(Kp=1.5, Ti=2.0, Td=0.1, dt=0.01,
                              u_min=-100, u_max=100)
code = gen_pid(dpid, class_name="MyPID", cfg=CodegenConfig(add_comments=False))

# Datei speichern
with open("pid_controller.py", "w") as f:
    f.write(code)
```

**Beispiel des generierten Codes:**

```python
# Generiert von pylabb.codegen für ESP32 / RP2040 / STM32
# Kp=1.5  Ti=2.0  Td=0.1  dt=0.01s

import time

class MyPID:
    # Zeitdiskreter PID-Regler ...

    def __init__(self):
        self.Kp = 1.5
        self.Ti = 2.0
        self.Td = 0.1
        self.dt = 0.01
        self.N  = 20.0
        self.u_min = -100.0
        self.u_max = 100.0
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_deriv = 0.0

    def reset(self):
        ...

    def update(self, setpoint, measurement):
        ...
```

### Digitalen Filter generieren (`gen_digital_filter`)

```python
from pylabb.codegen.micropython import gen_digital_filter
from pylabb.core.transfer_function import TransferFunction

# Diskrete ÜTF (z.B. durch Diskretisierung der kontinuierlichen ÜTF)
G = TransferFunction([1], [1, 2, 1])
Gd = G.discretize(dt=0.01, method='tustin')

code = gen_digital_filter(Gd, class_name="LowPassFilter")
```

### Zustandsraumregler generieren (`gen_state_feedback`)

```python
from pylabb.codegen.micropython import gen_state_feedback
from pylabb.core.state_space import StateSpace
import numpy as np

sys = StateSpace([[-2, 1],[0, -3]], [[0],[1]], [[1,0]], [[0]])
K = np.array([[1.5, 0.8]])  # Rückführvektor

code = gen_state_feedback(sys, K, class_name="StateFeedback")
```

### Vollständigen Regelkreis generieren (`gen_control_loop`)

```python
from pylabb.codegen.micropython import gen_control_loop
from pylabb.control.pid import DiscretePIDController

dpid = DiscretePIDController(Kp=2.0, Ti=1.0, Td=0.05, dt=0.01)
code = gen_control_loop(dpid, loop_period_ms=10,
                         cfg=CodegenConfig(include_uart=True,
                                           include_watchdog=True))

with open("main.py", "w") as f:
    f.write(code)
```

Der generierte `main.py` enthält sofort eine lauffähige Hauptschleife mit Timing, optionaler UART-Ausgabe und Watchdog-Reset.

### Deployment auf das Gerät

```bash
# mit mpremote (offizielles MicroPython-Tool)
pip install mpremote

mpremote connect /dev/ttyUSB0 fs cp pid_controller.py :pid_controller.py
mpremote connect /dev/ttyUSB0 fs cp main.py :main.py
mpremote connect /dev/ttyUSB0 run main.py
```

---

## 10. GUI – Grafische Benutzeroberfläche

### Start

```bash
pylabb-gui
```

### Layout

Das Hauptfenster ist in vier Bereiche unterteilt:

```
┌────────────────────────────────────────────────────────────────────┐
│ Menüleiste: Datei | Analyse | Codegen | Verifikation | Hilfe       │
├──────────────┬────────────────────────────────┬────────────────────┤
│              │  Plot-Tabs                     │  Verifikations-    │
│  System-     │   • Bode-Diagramm              │  Dock              │
│  Editor      │   • Nyquist-Ortskurve          │  (rechts)          │
│  (Dock)      │   • Sprungantwort              │                    │
│              │   • Pol-Nullstellen-Plan       │                    │
├──────────────┼────────────────────────────────┤                    │
│  Analyse-    │  Codegen-Panel                 │                    │
│  Panel       │                                │                    │
│  (Dock)      │                                │                    │
└──────────────┴────────────────────────────────┴────────────────────┘
│ Statusleiste                                                       │
└────────────────────────────────────────────────────────────────────┘
```

### System-Editor (linkes Dock)

- **Übertragungsfunktion definieren**: Zähler- und Nennerkoeffizienten eingeben (kommagetrennt, höchste Potenz zuerst)
- **PID-Parameter einstellen**: Kp, Ti, Td, N über Eingabefelder
- **Schaltfläche „Aktualisieren"**: Erzeugt alle Diagramme und Kennwerte neu

### Analyse-Panel (unteres Dock)

Zeigt automatisch berechnete Kennwerte an:
- Stabilitätseigenschaften (Phasen- und Amplitudenrand)
- Sprungantwort-Kennwerte (Anstiegszeit, Ausregelzeit, Überschwingen)
- Pol- und Nullstellen-Liste

### Plot-Tabs (Hauptbereich)

| Tab | Inhalt |
|-----|--------|
| Bode | Amplitudengang [dB] und Phasengang [°] über Frequenz |
| Nyquist | Ortskurve von $G(j\omega)$ in der komplexen Ebene |
| Sprungantwort | Führungs- und Störantwort mit Kenngrößen-Markierungen |
| Pol-Nullstellen | Pol-Nullstellen-Plan mit Stabilitätsgrenze |

### Codegen-Panel

- Auswahl des Zieltyps: PID / Digitaler Filter / Zustandsrückkopplung / Vollständiger Regelkreis
- Konfiguration: Plattform, Abtastzeit, UART, Watchdog
- **Generieren**: Erzeugt den Code im Vorschaufenster
- **Speichern**: Export in `.py`-Datei

### Verifikations-Dock (neu in 2.0.0)

Das **Regelkreis-Verifikations-Dock** (rechtes Seitenpanel, Menü
**Verifikation → Regelkreis-Äquivalenz prüfen**, Taste `F7`) ermöglicht den
grafischen Vergleich zweier Regelkreise:

- **Regelkreis A / B**: Je ein Formularblock für Strecke G(s) und Regler C(s)
  (Zähler/Nenner-Koeffizienten kommagetrennt)
- **Schaltfläche „Äquivalenz prüfen"**: Startet den Subgraph Algorithmus
- **Entscheidungs-Badge**: Farbiges Label zeigt das Ergebnis
  - Grün: Äquivalent
  - Blau: A enthält B
  - Lila: B enthält A
  - Orange: Strukturell verschieden
- **Adjazenzmatrizen A und B**: Tabellarische Darstellung (grün = Kante vorhanden)
- **Textuelle Zusammenfassung**: Entscheidung, Matrizengrößen, Kantenanzahl
- **„Aktuellen Regelkreis als A übernehmen"**: Synchronisiert Dock mit System-Editor

Der aktuelle Regelkreis aus dem System-Editor wird bei jeder Änderung automatisch
in Slot A des Verifikations-Docks übertragen.

---

## 11. Vollständige Beispiele

### Beispiel 1: PID-Entwurf für eine PT2-Strecke

```python
from pylabb.core.transfer_function import TransferFunction
from pylabb.control.pid import PIDController, ziegler_nichols_step
from pylabb.control.stability import analyze_stability
from pylabb.simulation.time_domain import ClosedLoopSimulator
from pylabb.core.signals import step_signal
import matplotlib.pyplot as plt

# PT2-Strecke: G(s) = 1 / (s² + 0.5s + 1)
G = TransferFunction([1.0], [1.0, 0.5, 1.0], name="PT2")

# Stabilitätsanalyse des offenen Kreises (nur Strecke)
info_ol = analyze_stability(G)
print("Offener Kreis:", info_ol)

# PID-Entwurf nach Ziegler-Nichols (FOPDT-Identifikation nötig)
# Hier direkt mit praxisnahen Werten:
pid = PIDController(Kp=2.0, Ti=4.0, Td=0.5)
C   = pid.transfer_function()

# Stabilitätsanalyse des offenen Kreises C·G
L = C * G
info_cl = analyze_stability(L)
print("Geschlossener Kreis:", info_cl)

# Simulation
dt = 0.01
w  = step_signal(t_end=20.0, dt=dt)
sim = ClosedLoopSimulator(plant=G, controller=C)
result = sim.simulate(setpoint=w)

print(f"Anstiegszeit:  {result.rise_time():.2f} s")
print(f"Überschwingen: {result.overshoot():.1f} %")
print(f"Ausregelzeit:  {result.settling_time():.2f} s")

plt.figure(figsize=(10, 4))
plt.plot(result.t, result.setpoint, 'k--', label='Sollwert w')
plt.plot(result.t, result.y, label='Ausgang y')
plt.xlabel("Zeit [s]")
plt.ylabel("Amplitude")
plt.title("Führungsverhalten PT2 mit PID")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

### Beispiel 2: LQR-Entwurf für ein Doppelintegrator-System

```python
import numpy as np
from pylabb.core.state_space import StateSpace
from pylabb.simulation.time_domain import simulate_ss

# Doppelintegrator: ẍ = u
A = [[0, 1], [0, 0]]
B = [[0], [1]]
C = [[1, 0]]
D = [[0]]

sys = StateSpace(A, B, C, D)

print("Steuerbar:", sys.is_controllable())

# LQR: Lage stark gewichten, Energie wenig
Q = np.diag([100, 1])
R = np.array([[1]])
K, S, eig = sys.lqr(Q, R)
print("LQR-Gain:", K)
print("Eigenwerte des geregelten Systems:", eig)
```

---

### Beispiel 3: MicroPython-Code für ESP32 generieren

```python
from pylabb.control.pid import DiscretePIDController
from pylabb.codegen.micropython import gen_control_loop, CodegenConfig

# Regler für Temperaturregelung (z.B. ESP32 mit PT100-Sensor)
dpid = DiscretePIDController(
    Kp    = 5.0,     # Verstärkung
    Ti    = 30.0,    # Nachstellzeit [s] – langsame Thermik
    Td    = 2.0,     # Vorhaltzeit
    dt    = 0.1,     # 100 ms Abtastzeit
    u_min = 0.0,     # Heizleistung min
    u_max = 100.0,   # Heizleistung max (%)
)

cfg = CodegenConfig(
    target           = "ESP32",
    tick_source      = "time.ticks_ms",
    include_uart     = True,    # Ausgabe über serielle Schnittstelle
    include_watchdog = True,    # Sicherheitsfunktion
    add_comments     = True,
)

code = gen_control_loop(dpid, loop_period_ms=100, cfg=cfg)

with open("esp32_temperature_control.py", "w") as f:
    f.write(code)

print("Code generiert – bereit für Upload via mpremote.")
```

---

### Beispiel 4: Bode-Diagramm mehrerer Systeme vergleichen

```python
from pylabb.core.transfer_function import TransferFunction
from pylabb.visualization.bode import bode_plot
import matplotlib.pyplot as plt

# Verschiedene Regler-Varianten
G   = TransferFunction([1], [1, 2, 1])  # Strecke
C1  = TransferFunction([2, 1], [1])     # P-Regler
C2  = TransferFunction([3, 2], [1, 0])  # PI-Regler

L1 = C1 * G   # Offener Kreis 1
L2 = C2 * G   # Offener Kreis 2

fig = bode_plot([L1, L2],
                labels=["P-Regler", "PI-Regler"],
                omega_range=(0.01, 100),
                title="Vergleich offener Kreis")
plt.show()
```

---

## 12. Tests und Code Coverage

```bash
# Alle Tests ausführen (mit Coverage-Report)
pytest

# Coverage-HTML-Report anzeigen
xdg-open doc/coverage/index.html     # Linux
# open doc/coverage/index.html       # macOS
```

Der Coverage-Report wird automatisch nach `doc/coverage/` geschrieben.

Aktuelle Testabdeckung (Version 2.1.0):

| Modul | Abdeckung |
|---|---|
| `core/transfer_function.py` | 99 % |
| `core/state_space.py` | 99 % |
| `core/signals.py` | 100 % |
| `core/math_utils.py` | 100 % |
| `control/pid.py` | 100 % |
| `control/design.py` | 100 % |
| `control/stability.py` | 100 % |
| `control/verification.py` | **100 %** |
| `control/bio_classify.py` | **100 %** |
| `simulation/time_domain.py` | 100 % |
| `visualization/*.py` | 100 % |
| `codegen/micropython.py` | 100 % |
| `gui/widgets/verification_widget.py` | **100 %** |
| **Gesamt** | **99 %** |

Einzelne Testdateien:

```bash
pytest tests/test_control.py -v
pytest tests/test_codegen.py -v
pytest tests/test_simulation.py -v
pytest tests/test_gui.py -v
pytest tests/test_verification.py -v
pytest tests/test_verification_widget.py -v
```

---

## 13. Projektstruktur

```
pylabb/
├── pyproject.toml           – Paketmetadaten und Build-Konfiguration
├── README.md                – Kurzübersicht
├── doc/
│   ├── MANUAL.md            – Dieses Handbuch
│   └── coverage/            – HTML-Coverage-Report (nach pytest)
├── src/
│   └── pylabb/
│       ├── __init__.py
│       ├── core/            – Grundlegende Datentypen
│       ├── control/         – Reglerentwurf, Stabilitäts- und Äquivalenzanalyse
│       ├── simulation/      – Zeitbereichssimulation
│       ├── visualization/   – matplotlib-Diagramme
│       ├── codegen/         – MicroPython-Codegenerator
│       └── gui/             – PyQt6-Benutzeroberfläche
└── tests/                   – pytest-Testsuites (99 % Coverage)
```

---

*PyLab Anwenderhandbuch – Version 2.1.0 – Stephan Epp*

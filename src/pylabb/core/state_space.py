"""
pylabb.core.state_space
======================
Zustandsraumdarstellung  ẋ = A·x + B·u,  y = C·x + D·u.

Unterstützt:
- Pole- / Eigenwert-Analyse
- Steuerbarkeit / Beobachtbarkeit (Grammsche Matrizen und Rang)
- Zeitantworten (Sprung, Impuls, beliebiger Eingang)
- Konvertierung in Übertragungsfunktion
- Diskretisierung per ZOH oder Euler
- Zustandsrückführung (Polvorgabe via scipy.signal.place_poles)
- LQR-Entwurf (scipy.linalg.solve_continuous_are)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
import scipy.signal as sig

if TYPE_CHECKING:
    from pylabb.core.transfer_function import TransferFunction


class StateSpace:
    """Lineare, zeitinvariante Zustandsraumdarstellung.

    Parameters
    ----------
    A  : Systemmatrix       (n×n).
    B  : Eingangsmatrix     (n×m).
    C  : Ausgangsmatrix     (p×n).
    D  : Durchgangsmatrix   (p×m).
    dt : Abtastzeit [s]; None → zeitkontinuierlich.
    name : Bezeichner.
    """

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
        dt: Optional[float] = None,
        name: str = "SS",
    ) -> None:
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.C = np.atleast_2d(np.asarray(C, dtype=float))
        self.D = np.atleast_2d(np.asarray(D, dtype=float))
        self.dt = dt
        self.name = name
        self._validate()

    # ------------------------------------------------------------------
    # Validierung
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError(f"A muss quadratisch sein, hat Form {self.A.shape}.")
        if self.B.shape[0] != n:
            raise ValueError("B-Zeilen müssen A-Ordnung entsprechen.")
        if self.C.shape[1] != n:
            raise ValueError("C-Spalten müssen A-Ordnung entsprechen.")
        if self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D hat falsche Dimension.")

    # ------------------------------------------------------------------
    # Eigenschaften
    # ------------------------------------------------------------------

    @property
    def order(self) -> int:
        """Systemordnung n."""
        return self.A.shape[0]

    @property
    def n_inputs(self) -> int:
        """Anzahl Eingänge m."""
        return self.B.shape[1]

    @property
    def n_outputs(self) -> int:
        """Anzahl Ausgänge p."""
        return self.C.shape[0]

    # ------------------------------------------------------------------
    # Stabilität / Analyse
    # ------------------------------------------------------------------

    def eigenvalues(self) -> NDArray:
        """Eigenwerte der Systemmatrix A."""
        return np.linalg.eigvals(self.A)

    def is_stable(self) -> bool:
        """Überprüft Stabilität anhand der Eigenwerte."""
        ev = self.eigenvalues()
        if self.dt is None:
            return bool(np.all(np.real(ev) < 0))
        return bool(np.all(np.abs(ev) < 1.0))

    def controllability_matrix(self) -> NDArray:
        """Steuerbarkeitsmatrix R = [B, AB, A²B, …, A^{n-1}B]."""
        n = self.order
        M = self.B.copy()
        Ak = np.eye(n)
        for _ in range(1, n):
            Ak = Ak @ self.A
            M = np.hstack([M, Ak @ self.B])
        return M

    def observability_matrix(self) -> NDArray:
        """Beobachtbarkeitsmatrix O = [C; CA; CA²; …; CA^{n-1}]ᵀ."""
        n = self.order
        M = self.C.copy()
        Ak = np.eye(n)
        for _ in range(1, n):
            Ak = Ak @ self.A
            M = np.vstack([M, self.C @ Ak])
        return M

    def is_controllable(self) -> bool:
        """True, wenn der Rangtest der Steuerbarkeitsmatrix vollen Rang zeigt."""
        Mc = self.controllability_matrix()
        return int(np.linalg.matrix_rank(Mc)) == self.order

    def is_observable(self) -> bool:
        """True, wenn der Rangtest der Beobachtbarkeitsmatrix vollen Rang zeigt."""
        Mo = self.observability_matrix()
        return int(np.linalg.matrix_rank(Mo)) == self.order

    # ------------------------------------------------------------------
    # Freqenzgang
    # ------------------------------------------------------------------

    def freqresp(
        self, omega: Optional[NDArray] = None, n_points: int = 500
    ) -> tuple[NDArray, NDArray]:
        """Frequenzgang H(jω) = C·(jωI−A)⁻¹·B + D."""
        if omega is None:
            omega = np.logspace(-2, 2, n_points)
        n = self.order
        H_list = []
        for w in omega:
            if self.dt is None:
                jw = 1j * w
            else:
                jw = np.exp(1j * w * self.dt)
            mat = jw * np.eye(n) - self.A
            H_list.append(self.C @ np.linalg.solve(mat, self.B) + self.D)
        H = np.array(H_list)  # (n_omega, p, m)
        return omega, H

    # ------------------------------------------------------------------
    # Zeitantworten
    # ------------------------------------------------------------------

    def step_response(
        self, t_end: Optional[float] = None, dt: Optional[float] = None
    ) -> tuple[NDArray, NDArray]:
        """Sprungantwort.

        Returns
        -------
        t, y  (y hat Form (n_t, p, m))
        """
        t_arr = self._sim_time(t_end, dt)
        if self.dt is None:
            t_out, y_out = sig.step(
                sig.lti(self.A, self.B, self.C, self.D), T=t_arr
            )
        else:
            t_out, y_out = sig.dstep(
                sig.dlti(self.A, self.B, self.C, self.D, dt=self.dt),
                t=np.arange(int(t_arr[-1] / self.dt) + 1) * self.dt,
            )
            y_out = np.squeeze(y_out)
        return t_out, y_out

    def lsim(self, u: NDArray, t: NDArray) -> tuple[NDArray, NDArray]:
        """Lineare Simulation mit beliebigem Eingang.

        Returns
        -------
        t, y
        """
        if self.dt is None:
            t_out, y_out, _ = sig.lsim(
                sig.lti(self.A, self.B, self.C, self.D), U=u, T=t
            )
        else:
            t_out, y_out, _ = sig.dlsim(
                sig.dlti(self.A, self.B, self.C, self.D, dt=self.dt),
                u=u.reshape(-1, 1), t=t,
            )
            y_out = y_out.flatten()
        return t_out, y_out

    # ------------------------------------------------------------------
    # Reglerentwurf
    # ------------------------------------------------------------------

    def place_poles(
        self, desired_poles: Sequence[complex]
    ) -> NDArray:
        """Zustandsrückführung K durch Polvorgabe (nur SISO/MISO).

        Returns
        -------
        K : Rückführungsmatrix (m×n) so dass  eig(A − B·K) ≈ desired_poles.
        """
        if not self.is_controllable():
            raise ValueError("System ist nicht vollständig steuerbar.")
        result = sig.place_poles(self.A, self.B, desired_poles)
        return result.gain_matrix

    def lqr(
        self,
        Q: Optional[NDArray] = None,
        R: Optional[NDArray] = None,
    ) -> tuple[NDArray, NDArray]:
        """LQR-Reglerentwurf (Linear Quadratic Regulator).

        Minimiert J = ∫(xᵀQx + uᵀRu) dt durch Lösen der
        algebraischen Riccati-Gleichung.

        Parameters
        ----------
        Q : Zustandswichtungsmatrix (n×n), Standard: Einheitsmatrix.
        R : Eingangswichtungsmatrix (m×m), Standard: Einheitsmatrix.

        Returns
        -------
        K  : Optimale Verstärkungsmatrix (m×n).
        P  : Lösung der Riccati-Gleichung.
        """
        n = self.order
        m = self.n_inputs
        Q = Q if Q is not None else np.eye(n)
        R = R if R is not None else np.eye(m)
        P = la.solve_continuous_are(self.A, self.B, Q, R)
        K = np.linalg.solve(R, self.B.T @ P)
        return K, P

    def observer_gain(
        self, desired_poles: Sequence[complex]
    ) -> NDArray:
        """Beobachter-Rückführung L per Polvorgabe (Luenberger-Beobachter).

        Returns
        -------
        L : Beobachtermatrix (n×p).
        """
        if not self.is_observable():
            raise ValueError("System ist nicht vollständig beobachtbar.")
        # Dualitätsprinzip: Beobachter ↔ Regler des dualen Systems
        result = sig.place_poles(self.A.T, self.C.T, desired_poles)
        return result.gain_matrix.T

    # ------------------------------------------------------------------
    # Konvertierung
    # ------------------------------------------------------------------

    def to_transfer_function(self) -> "TransferFunction":
        """Konvertiert in Übertragungsfunktion (nur SISO)."""
        from pylabb.core.transfer_function import TransferFunction as _TF
        num, den = sig.ss2tf(self.A, self.B, self.C, self.D)
        return _TF(num[0], den, dt=self.dt, name=self.name)

    def discretize(
        self, dt: float, method: str = "zoh"
    ) -> "StateSpace":
        """Diskretisiert das kontinuierliche System.

        Parameters
        ----------
        dt     : Abtastzeit [s].
        method : 'zoh'   – Zero-Order-Hold (Standard)
                 'euler' – Euler-Vorwärtsintegration
                 'bilinear' – Tustin
        """
        if self.dt is not None:
            raise ValueError("System ist bereits zeitdiskret.")
        sys_d = sig.cont2discrete(
            (self.A, self.B, self.C, self.D), dt=dt, method=method
        )
        Ad, Bd, Cd, Dd = sys_d[:4]
        return StateSpace(Ad, Bd, Cd, Dd, dt=dt, name=self.name + "_d")

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _sim_time(
        self,
        t_end: Optional[float],
        dt: Optional[float],
    ) -> NDArray:
        ev = np.abs(np.real(self.eigenvalues()))
        ev_pos = ev[ev > 0]
        tau = 1.0 / np.min(ev_pos) if len(ev_pos) else 5.0
        t_end = t_end or min(max(10 * tau, 1.0), 200.0)
        dt = dt or t_end / 2000
        return np.linspace(0, t_end, int(t_end / dt) + 1)

    def __repr__(self) -> str:
        return (
            f"StateSpace(n={self.order}, m={self.n_inputs}, p={self.n_outputs}, "
            f"stable={self.is_stable()}, dt={self.dt}, name={self.name!r})"
        )


# ---------------------------------------------------------------------------
# Typ-Alias für Import-Kompatibilität
# ---------------------------------------------------------------------------
ArrayLike = NDArray  # wird oben als  from numpy.typing import NDArray  definiert

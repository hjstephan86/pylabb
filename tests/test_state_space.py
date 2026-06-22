"""
Tests für pylabb.core.state_space – 100 % Pfadabdeckung
"""
import numpy as np
import pytest

from pylabb.core.state_space import StateSpace


def _double_integrator():
    """ẍ = u  →  A=[[0,1],[0,0]], B=[[0],[1]], C=[[1,0]], D=[[0]]"""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    return StateSpace(A, B, C, D)


def _pt1_ss():
    """PT1: ẋ = -x + u,  y = x"""
    return StateSpace([[-1]], [[1]], [[1]], [[0]])


def _pt2_ss():
    """PT2: ẍ + 1.4ẋ + x = u"""
    A = np.array([[0.0, 1.0], [-1.0, -1.4]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    return StateSpace(A, B, C, D)


class TestStateSpaceCreation:
    def test_basic_creation(self):
        ss = _pt1_ss()
        assert ss.order == 1
        assert ss.n_inputs == 1
        assert ss.n_outputs == 1

    def test_repr(self):
        ss = _pt1_ss()
        r = repr(ss)
        assert "StateSpace" in r
        assert "stable=True" in r

    def test_validation_A_not_square(self):
        with pytest.raises(ValueError):
            StateSpace([[1, 2, 3], [4, 5, 6]], [[1], [1]], [[1, 1]], [[0]])

    def test_validation_B_row_mismatch(self):
        A = np.eye(2)
        B = np.ones((3, 1))
        C = np.ones((1, 2))
        D = np.zeros((1, 1))
        with pytest.raises(ValueError):
            StateSpace(A, B, C, D)

    def test_validation_C_col_mismatch(self):
        A = np.eye(2)
        B = np.ones((2, 1))
        C = np.ones((1, 3))
        D = np.zeros((1, 1))
        with pytest.raises(ValueError):
            StateSpace(A, B, C, D)

    def test_validation_D_shape(self):
        A = np.eye(2)
        B = np.ones((2, 1))
        C = np.ones((1, 2))
        D = np.zeros((2, 2))  # should be (1,1)
        with pytest.raises(ValueError):
            StateSpace(A, B, C, D)


class TestStabilityAnalysis:
    def test_stable_pt1(self):
        ss = _pt1_ss()
        assert ss.is_stable()

    def test_unstable(self):
        ss = StateSpace([[1]], [[1]], [[1]], [[0]])
        assert not ss.is_stable()

    def test_eigenvalues_pt2(self):
        ss = _pt2_ss()
        ev = ss.eigenvalues()
        assert len(ev) == 2
        assert np.all(np.real(ev) < 0)

    def test_discrete_stable(self):
        # Eigenvalues inside unit circle
        ss_d = StateSpace([[0.5]], [[0.1]], [[1]], [[0]], dt=0.01)
        assert ss_d.is_stable()

    def test_discrete_unstable(self):
        ss_d = StateSpace([[1.5]], [[0.1]], [[1]], [[0]], dt=0.01)
        assert not ss_d.is_stable()


class TestControllabilityObservability:
    def test_controllable(self):
        ss = _pt2_ss()
        assert ss.is_controllable()

    def test_observable(self):
        ss = _pt2_ss()
        assert ss.is_observable()

    def test_controllability_matrix_shape(self):
        ss = _pt2_ss()
        R = ss.controllability_matrix()
        assert R.shape == (2, 2)

    def test_observability_matrix_shape(self):
        ss = _pt2_ss()
        O = ss.observability_matrix()
        assert O.shape == (2, 2)

    def test_not_controllable(self):
        # Uncontrollable: B = 0
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.array([[0.0], [0.0]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])
        ss = StateSpace(A, B, C, D)
        assert not ss.is_controllable()


class TestFreqResponse:
    def test_freqresp_shape(self):
        ss = _pt1_ss()
        omega, H = ss.freqresp(n_points=100)
        assert len(omega) == 100
        assert len(H) == 100

    def test_freqresp_custom_omega(self):
        ss = _pt1_ss()
        omega_in = np.logspace(-1, 1, 50)
        omega_out, H = ss.freqresp(omega=omega_in)
        assert len(omega_out) == 50

    def test_dc_gain_pt1(self):
        ss = _pt1_ss()
        omega_dc = np.array([1e-6])
        _, H = ss.freqresp(omega=omega_dc)
        assert abs(abs(H[0, 0, 0]) - 1.0) < 0.01

    def test_freqresp_discrete(self):
        ss_d = _pt1_ss().discretize(dt=0.01)
        omega, H = ss_d.freqresp(n_points=50)
        assert len(H) == 50


class TestTimeResponses:
    def test_step_response_shape(self):
        ss = _pt1_ss()
        t, y = ss.step_response(t_end=10.0)
        assert len(t) > 0
        assert len(y) > 0

    def test_step_response_final_value(self):
        ss = _pt1_ss()
        t, y = ss.step_response(t_end=20.0)
        assert abs(y[-1] - 1.0) < 0.05

    def test_step_response_discrete(self):
        ss_d = _pt1_ss().discretize(dt=0.01)
        t, y = ss_d.step_response(t_end=10.0)
        assert len(t) > 0

    def test_lsim_continuous(self):
        ss = _pt1_ss()
        t = np.linspace(0, 10, 1000)
        u = np.ones(1000)
        t_out, y_out = ss.lsim(u, t)
        assert len(t_out) == len(t)
        assert abs(y_out[-1] - 1.0) < 0.05

    def test_lsim_discrete(self):
        ss_d = _pt1_ss().discretize(dt=0.01)
        t = np.arange(500) * 0.01
        u = np.ones(500)
        t_out, y_out = ss_d.lsim(u, t)
        assert len(y_out) > 0


class TestControllerDesign:
    def test_place_poles(self):
        ss = _pt2_ss()
        desired = [-2.0, -3.0]
        K = ss.place_poles(desired)
        # Check closed-loop poles are roughly at desired locations
        A_cl = ss.A - ss.B @ K
        cl_eig = np.linalg.eigvals(A_cl)
        for d in desired:
            assert any(abs(ev - d) < 0.1 for ev in cl_eig)

    def test_place_poles_uncontrollable(self):
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.zeros((2, 1))
        ss = StateSpace(A, B, np.eye(2), np.zeros((2, 1)))
        with pytest.raises(ValueError):
            ss.place_poles([-3.0, -4.0])

    def test_lqr(self):
        ss = _pt2_ss()
        K, P = ss.lqr()
        assert K.shape == (1, 2)
        assert P.shape == (2, 2)
        # P should be positive definite
        assert np.all(np.linalg.eigvals(P) > 0)

    def test_lqr_custom_QR(self):
        ss = _pt2_ss()
        Q = 10 * np.eye(2)
        R = 0.1 * np.eye(1)
        K, P = ss.lqr(Q=Q, R=R)
        assert K.shape == (1, 2)

    def test_observer_gain(self):
        ss = _pt2_ss()
        desired = [-10.0, -12.0]
        L = ss.observer_gain(desired)
        assert L.shape == (2, 1)

    def test_observer_gain_unobservable(self):
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.ones((2, 1))
        C = np.zeros((1, 2))  # unobservable: output matrix = 0
        ss = StateSpace(A, B, C, np.zeros((1, 1)))
        with pytest.raises(ValueError):
            ss.observer_gain([-5.0, -6.0])


class TestConversionAndDiscretization:
    def test_to_transfer_function(self):
        ss = _pt1_ss()
        tf = ss.to_transfer_function()
        assert abs(tf.dc_gain() - 1.0) < 0.01

    def test_discretize_zoh(self):
        ss = _pt1_ss()
        ss_d = ss.discretize(dt=0.01, method="zoh")
        assert ss_d.dt == 0.01
        assert ss_d.order == 1

    def test_discretize_euler(self):
        ss = _pt1_ss()
        ss_d = ss.discretize(dt=0.01, method="euler")
        assert ss_d.dt == 0.01

    def test_discretize_bilinear(self):
        ss = _pt1_ss()
        ss_d = ss.discretize(dt=0.01, method="bilinear")
        assert ss_d.dt == 0.01

    def test_discretize_already_discrete_raises(self):
        ss_d = _pt1_ss().discretize(dt=0.01)
        with pytest.raises(ValueError):
            ss_d.discretize(dt=0.01)

    def test_repr_discrete(self):
        ss_d = _pt1_ss().discretize(dt=0.01)
        r = repr(ss_d)
        assert "dt=0.01" in r

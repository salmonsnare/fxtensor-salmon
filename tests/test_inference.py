"""Tests for bayesian_inversion, conditional and is_deterministic.

References: K. Cho and B. Jacobs, "Disintegration and Bayesian Inversion via
String Diagrams" (arXiv:1709.00322); T. Fritz (arXiv:1908.07021).
"""
import numpy as np
import pytest

from fxtensor_salmon import FXTensor

from test_axioms import deterministic_kernel, random_kernel

HEALTH = ["Sick", "Healthy"]
RESULT = ["Positive", "Negative"]


@pytest.fixture
def prior():
    return FXTensor([[], [HEALTH]], data=np.array([0.01, 0.99]))


@pytest.fixture
def test_kernel():
    return FXTensor([[HEALTH], [RESULT]], data=np.array([[0.9, 0.1], [0.05, 0.95]]))


def joint_forward(prior, f):
    """prior ; copy_X ; (id_X ⊗ f)  :  I → X ⊗ Y"""
    x = prior.labels[1] or prior.profile[1]
    return prior.composition(FXTensor.copy_tensor(x)).composition(
        FXTensor.identity_tensor(x).tensor_product(f)
    )


def joint_backward(prior, f, f_dagger):
    """prior ; f ; copy_Y ; (f† ⊗ id_Y)  :  I → X ⊗ Y"""
    y = f.labels[1] or f.profile[1]
    return prior.composition(f).composition(FXTensor.copy_tensor(y)).composition(
        f_dagger.tensor_product(FXTensor.identity_tensor(y))
    )


class TestBayesianInversion:
    def test_hand_computed(self, prior, test_kernel):
        inv = test_kernel.bayesian_inversion(prior)
        assert inv.labels == ([RESULT], [HEALTH])
        pos = inv.get_label_index(0, "Positive")
        sick = inv.get_label_index(1, "Sick")
        assert np.isclose(inv.data[pos, sick], 0.009 / 0.0585)  # = 2/13
        assert np.isclose(inv.data[1, 0], 0.001 / 0.9415)
        assert inv.is_markov()

    def test_joint_equation_labeled(self, prior, test_kernel):
        inv = test_kernel.bayesian_inversion(prior)
        lhs = joint_forward(prior, test_kernel)
        rhs = joint_backward(prior, test_kernel, inv)
        assert lhs.labels == (None, [HEALTH, RESULT])
        assert lhs == rhs

    @pytest.mark.parametrize("x,y", [([2], [3]), ([2, 3], [2]), ([3], [2, 2])])
    def test_joint_equation_random(self, x, y):
        p = random_kernel([], x, 10)
        f = random_kernel(x, y, 11)
        inv = f.bayesian_inversion(p)
        assert inv.profile == [y, x]
        assert inv.is_markov()
        assert joint_forward(p, f) == joint_backward(p, f, inv)

    def test_zero_evidence_rows_stay_zero(self):
        p = FXTensor([[], [2]], data=np.array([0.5, 0.5]))
        f = FXTensor([[2], [3]], data=np.array([[0.5, 0.5, 0.0], [0.2, 0.8, 0.0]]))
        inv = f.bayesian_inversion(p)
        assert np.allclose(inv.data[2], 0.0)
        assert inv.is_markov()
        assert joint_forward(p, f) == joint_backward(p, f, inv)

    def test_double_inversion_recovers_kernel_on_support(self):
        p = random_kernel([], [3], 12)
        f = random_kernel([3], [2], 13)
        inv = f.bayesian_inversion(p)
        assert inv.bayesian_inversion(p.composition(f)) == f

    def test_labels_from_prior_when_kernel_unlabeled(self, prior):
        f = FXTensor([[2], [2]], data=np.array([[0.9, 0.1], [0.05, 0.95]]))
        inv = f.bayesian_inversion(prior)
        assert inv.labels == (None, None)

    def test_rejects_non_state_prior(self, test_kernel):
        with pytest.raises(ValueError):
            test_kernel.bayesian_inversion(test_kernel)

    def test_rejects_mismatched_prior(self, test_kernel):
        with pytest.raises(ValueError):
            test_kernel.bayesian_inversion(FXTensor([[], [3]], data=np.ones(3) / 3))


class TestConditional:
    def test_matches_conditionalization_on_states(self):
        s = random_kernel([], [2, 3, 2], 20)
        for k in (1, 2, 3):
            assert s.conditional(k) == s.conditionalization(k)

    def test_hand_computed_kernel(self):
        # f: A → X ⊗ Y with A = {a}, X = {x0, x1}, Y = {y0, y1}
        f = FXTensor(
            [[["a"]], [["x0", "x1"], ["y0", "y1"]]],
            data=np.array([[[0.1, 0.3], [0.2, 0.4]]]),
        )
        c = f.conditional(2)
        assert c.labels == ([["a"], ["x0", "x1"]], [["y0", "y1"]])
        assert c.profile == [[1, 2], [2]]
        assert np.allclose(c.data, [[[0.25, 0.75], [1 / 3, 2 / 3]]])
        assert c.is_markov()

    def test_disintegration_equation(self):
        # f(x, y | a) = f_X(x | a) * c(y | a, x)
        f = random_kernel([2], [3, 2], 21)
        f_x = f.marginalization(2)
        c = f.conditional(2)
        assert np.allclose(f.data, f_x.data[:, :, None] * c.data)

    def test_disintegration_diagram(self):
        # f = copy_A ; (id_A ⊗ f_X) ; (id_A ⊗ copy_X) ; (c ⊗ id_X) ; swap
        a, x, y = [2], [3], [2]
        f = random_kernel(a, x + y, 22)
        f_x = f.marginalization(2)
        c = f.conditional(2)
        ida = FXTensor.identity_tensor(a)
        rebuilt = (
            FXTensor.copy_tensor(a)
            .composition(ida.tensor_product(f_x))
            .composition(ida.tensor_product(FXTensor.copy_tensor(x)))
            .composition(c.tensor_product(FXTensor.identity_tensor(x)))
            .composition(FXTensor.swap(y, x))
        )
        assert rebuilt == f

    def test_zero_slice_stays_zero(self):
        f = FXTensor([[1], [2, 2]], data=np.array([[[0.5, 0.5], [0.0, 0.0]]]))
        c = f.conditional(2)
        assert np.allclose(c.data[0, 1], 0.0)
        assert c.is_markov()

    @pytest.mark.parametrize("index", [0, 3])
    def test_out_of_bounds(self, index):
        f = random_kernel([2], [2, 2], 23)
        with pytest.raises(ValueError):
            f.conditional(index)


class TestIsDeterministic:
    def test_one_hot_kernel(self):
        f = FXTensor([[["a", "b"]], [["x", "y", "z"]]], data=np.array([[0, 0, 1], [1, 0, 0]]))
        assert f.is_deterministic()

    def test_point_mass_state(self):
        assert FXTensor([[], [2, 2]], data=np.array([[0, 1], [0, 0]])).is_deterministic()
        assert not FXTensor([[], [2]], data=np.array([0.5, 0.5])).is_deterministic()

    def test_stochastic_kernel_is_not(self):
        assert not random_kernel([2], [3], 30).is_deterministic()

    def test_zero_row_is_not(self):
        assert not FXTensor([[2], [2]], data=np.array([[1.0, 0.0], [0.0, 0.0]])).is_deterministic()

    def test_composition_of_deterministic(self):
        f = deterministic_kernel([2], [3], 31)
        g = deterministic_kernel([3], [2, 2], 32)
        assert f.composition(g).is_deterministic()
        assert f.tensor_product(g).is_deterministic()

    def test_tolerance(self):
        assert FXTensor([[2], [2]], data=np.array([[1 - 1e-12, 1e-12], [0, 1]])).is_deterministic()

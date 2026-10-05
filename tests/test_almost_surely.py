"""Tests for support, almost_surely_equal and is_absolutely_continuous.

References: T. Fritz (arXiv:1908.07021), almost sure equality;
T. Fritz, T. Gonda, A. Lorenzin, P. Perrone, D. Stein, "Absolute continuity,
supports and idempotent splitting in categorical probability" (arXiv:2308.00651).
"""
import numpy as np
import pytest

from fxtensor_salmon import FXTensor

from test_axioms import random_kernel

X = ["x0", "x1", "x2"]
Y = ["y0", "y1"]


def joint(prior, f):
    """prior ; copy ; (id ⊗ f)"""
    x = prior.profile[1]
    return prior.composition(FXTensor.copy_tensor(x)).composition(
        FXTensor.identity_tensor(x).tensor_product(f)
    )


@pytest.fixture
def prior():
    # x2 has probability zero
    return FXTensor([[], [X]], data=np.array([0.4, 0.6, 0.0]))


@pytest.fixture
def f():
    return FXTensor([[X], [Y]], data=np.array([[0.2, 0.8], [0.5, 0.5], [1.0, 0.0]]))


@pytest.fixture
def g():
    # differs from f only on the null point x2
    return FXTensor([[X], [Y]], data=np.array([[0.2, 0.8], [0.5, 0.5], [0.3, 0.7]]))


class TestSupport:
    def test_state(self, prior):
        s = prior.support()
        assert s.labels == (None, [X])
        assert np.array_equal(s.data, [1.0, 1.0, 0.0])

    def test_kernel_relation(self, f):
        s = f.support()
        assert s.profile == f.profile
        assert s.labels == f.labels
        assert np.array_equal(s.data, [[1, 1], [1, 1], [1, 0]])

    def test_tolerance(self):
        p = FXTensor([[], [2]], data=np.array([1 - 1e-12, 1e-12]))
        assert np.array_equal(p.support().data, [1.0, 0.0])

    def test_support_of_composition(self, prior, f):
        # supp(p;f) = {y : exists x in supp(p) with f(y|x) > 0}
        expected = (prior.support().data @ f.support().data) > 0
        assert np.array_equal(prior.composition(f).support().data, expected.astype(float))


class TestAlmostSurelyEqual:
    def test_differs_only_on_null_set(self, prior, f, g):
        assert f.almost_surely_equal(g, prior)
        assert f != g

    def test_matches_joint_equation(self, prior, f, g):
        assert joint(prior, f) == joint(prior, g)

    def test_differs_on_support(self, prior, f):
        h = FXTensor([[X], [Y]], data=np.array([[0.3, 0.7], [0.5, 0.5], [1.0, 0.0]]))
        assert not f.almost_surely_equal(h, prior)
        assert joint(prior, f) != joint(prior, h)

    def test_full_support_prior_means_equality(self, f, g):
        uniform = FXTensor([[], [X]], data=np.full(3, 1 / 3))
        assert not f.almost_surely_equal(g, uniform)
        assert f.almost_surely_equal(f.copy(), uniform)

    def test_double_bayesian_inversion_is_almost_surely_equal(self, prior, g):
        # with a non-full-support prior, (g†)† differs from g on x2 but agrees a.s.
        inv = g.bayesian_inversion(prior)
        double = inv.bayesian_inversion(prior.composition(g))
        assert double != g
        assert double.almost_surely_equal(g, prior)

    def test_random_multi_axis(self):
        p = random_kernel([], [2, 3], 40)
        data = p.data.copy()
        data[1, 2] = 0.0
        p = FXTensor([[], [2, 3]], data=data / data.sum())
        f = random_kernel([2, 3], [2], 41)
        g_data = f.data.copy()
        g_data[1, 2] = [0.9, 0.1]
        g = FXTensor([[2, 3], [2]], data=g_data)
        assert f.almost_surely_equal(g, p)
        assert joint(p, f) == joint(p, g)

    def test_labels_are_ignored(self, prior, f):
        unlabeled = FXTensor([[3], [2]], data=f.data)
        assert f.almost_surely_equal(unlabeled, prior)

    def test_errors(self, prior, f):
        with pytest.raises(ValueError):
            f.almost_surely_equal(FXTensor([[3], [3]], data=np.eye(3)), prior)
        with pytest.raises(ValueError):
            f.almost_surely_equal(f, f)
        with pytest.raises(ValueError):
            f.almost_surely_equal(f, FXTensor([[], [2]], data=np.array([0.5, 0.5])))


class TestAbsoluteContinuity:
    def test_states(self):
        p = FXTensor([[], [3]], data=np.array([0.5, 0.5, 0.0]))
        q = FXTensor([[], [3]], data=np.array([0.2, 0.3, 0.5]))
        assert p.is_absolutely_continuous(q)
        assert not q.is_absolutely_continuous(p)

    def test_equivalent_to_support_inclusion(self):
        rng = np.random.default_rng(50)
        for _ in range(20):
            a = rng.random(4) * (rng.random(4) > 0.4)
            b = rng.random(4) * (rng.random(4) > 0.4)
            p, q = FXTensor([[], [4]], data=a), FXTensor([[], [4]], data=b)
            inclusion = np.all(p.support().data <= q.support().data)
            assert p.is_absolutely_continuous(q) == inclusion

    def test_kernels_inputwise(self, f, g):
        assert f.is_absolutely_continuous(g)
        assert not g.is_absolutely_continuous(f)

    def test_reflexive_and_transitive(self):
        p = FXTensor([[], [3]], data=np.array([1.0, 0.0, 0.0]))
        q = FXTensor([[], [3]], data=np.array([0.5, 0.5, 0.0]))
        r = FXTensor([[], [3]], data=np.full(3, 1 / 3))
        assert p.is_absolutely_continuous(p)
        assert p.is_absolutely_continuous(q) and q.is_absolutely_continuous(r)
        assert p.is_absolutely_continuous(r)

    def test_preserved_by_composition(self, f):
        p = FXTensor([[], [X]], data=np.array([1.0, 0.0, 0.0]))
        q = FXTensor([[], [X]], data=np.array([0.5, 0.0, 0.5]))
        assert p.is_absolutely_continuous(q)
        assert p.composition(f).is_absolutely_continuous(q.composition(f))

    def test_almost_sure_equality_transfers(self, prior, f, g):
        # if p ≪ q and f = g q-a.s., then f = g p-a.s.
        p = FXTensor([[], [X]], data=np.array([1.0, 0.0, 0.0]))
        assert p.is_absolutely_continuous(prior)
        assert f.almost_surely_equal(g, prior)
        assert f.almost_surely_equal(g, p)

    def test_profile_mismatch(self):
        with pytest.raises(ValueError):
            FXTensor([[], [2]], data=np.ones(2)).is_absolutely_continuous(FXTensor([[], [3]], data=np.ones(3)))

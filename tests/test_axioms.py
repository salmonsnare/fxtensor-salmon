"""Markov category axioms checked on random finite stochastic tensors.

References: T. Fritz, "A synthetic approach to Markov kernels, conditional
independence and theorems on sufficient statistics" (arXiv:1908.07021).
"""
import numpy as np
import pytest

from fxtensor_salmon import FXTensor

SHAPES = [([2], [3]), ([2], [2, 3]), ([2, 2], [3]), ([], [2, 2]), ([3], [])]


def random_kernel(dom, cod, seed):
    rng = np.random.default_rng(seed)
    data = rng.random(tuple(dom + cod)) + 0.05
    axes = tuple(range(len(dom), len(dom) + len(cod)))
    if axes:
        data = data / data.sum(axis=axes, keepdims=True)
    else:
        data = np.ones(tuple(dom))
    return FXTensor([list(dom), list(cod)], data=data)


def deterministic_kernel(dom, cod, seed):
    rng = np.random.default_rng(seed)
    data = np.zeros(tuple(dom + cod))
    cod_size = int(np.prod(cod))
    for idx in np.ndindex(*dom):
        target = np.unravel_index(rng.integers(cod_size), tuple(cod))
        data[idx + target] = 1
    return FXTensor([list(dom), list(cod)], data=data)


class TestTensorProductRegression:
    def test_kernel_with_longer_state_codomain(self):
        a = random_kernel([2], [2], 0)
        b = random_kernel([], [2, 3], 1)
        result = a.tensor_product(b)
        assert result.profile == [[2], [2, 2, 3]]
        assert np.allclose(result.data, np.einsum("ab,cd->abcd", a.data, b.data))

    @pytest.mark.parametrize("f_shape", SHAPES)
    @pytest.mark.parametrize("g_shape", SHAPES)
    def test_matches_einsum(self, f_shape, g_shape):
        f = random_kernel(*f_shape, 2)
        g = random_kernel(*g_shape, 3)
        fd, fc = len(f_shape[0]), len(f_shape[1])
        gd, gc = len(g_shape[0]), len(g_shape[1])
        letters = "abcdefghij"
        fi, gi = letters[:fd + fc], letters[fd + fc:fd + fc + gd + gc]
        out = fi[:fd] + gi[:gd] + fi[fd:] + gi[gd:]
        expected = np.einsum(f"{fi},{gi}->{out}", f.data, g.data)
        result = f.tensor_product(g)
        assert result.profile == [f_shape[0] + g_shape[0], f_shape[1] + g_shape[1]]
        assert np.allclose(result.data, expected)

    def test_labeled_mixed_lengths(self):
        f = FXTensor([[["a", "b"]], [["x", "y"]]], data=np.array([[0.2, 0.8], [0.6, 0.4]]))
        g = FXTensor([[], [["p", "q"], ["u", "v", "w"]]], data=np.full((2, 3), 1 / 6))
        result = f.tensor_product(g)
        assert result.labels == ([["a", "b"]], [["x", "y"], ["p", "q"], ["u", "v", "w"]])
        assert result.is_markov()

    def test_labeled_with_unlabeled_drops_labels(self):
        f = FXTensor([[["a", "b"]], [["x", "y"]]], data=np.array([[0.2, 0.8], [0.6, 0.4]]))
        g = FXTensor.identity_tensor([2])
        for result in (f.tensor_product(g), g.tensor_product(f)):
            assert result.profile == [[2, 2], [2, 2]]
            assert result.labels == (None, None)
        scalar = FXTensor([[], []], data=np.array(1.0))
        assert f.tensor_product(scalar).labels == f.labels


class TestMonoidalCategory:
    def test_composition_associative(self):
        f = random_kernel([2], [3], 0)
        g = random_kernel([3], [2, 2], 1)
        h = random_kernel([2, 2], [3], 2)
        assert f.composition(g).composition(h) == f.composition(g.composition(h))

    def test_identity_laws(self):
        f = random_kernel([2], [2, 3], 0)
        assert FXTensor.identity_tensor([2]).composition(f) == f
        assert f.composition(FXTensor.identity_tensor([2, 3])) == f

    def test_tensor_product_associative(self):
        f = random_kernel([2], [3], 0)
        g = random_kernel([], [2, 2], 1)
        h = random_kernel([3], [2], 2)
        assert f.tensor_product(g).tensor_product(h) == f.tensor_product(g.tensor_product(h))

    def test_interchange_law(self):
        f = random_kernel([2], [3], 0)
        h = random_kernel([3], [2, 2], 1)
        g = random_kernel([2, 3], [2], 2)
        k = random_kernel([2], [3, 2, 2], 3)
        lhs = f.tensor_product(g).composition(h.tensor_product(k))
        rhs = f.composition(h).tensor_product(g.composition(k))
        assert lhs == rhs

    def test_swap_natural(self):
        f = random_kernel([2], [3], 0)
        g = random_kernel([3], [2, 2], 1)
        lhs = f.tensor_product(g).composition(FXTensor.swap([3], [2, 2]))
        rhs = FXTensor.swap([2], [3]).composition(g.tensor_product(f))
        assert lhs == rhs

    def test_swap_involutive(self):
        assert FXTensor.swap([2], [3, 2]).composition(FXTensor.swap([3, 2], [2])) == FXTensor.identity_tensor([2, 3, 2])


@pytest.mark.parametrize("x", [[2], [3], [2, 3]])
class TestCopyComonoid:
    def test_cocommutative(self, x):
        copy = FXTensor.copy_tensor(x)
        assert copy.composition(FXTensor.swap(x, x)) == copy

    def test_coassociative(self, x):
        copy = FXTensor.copy_tensor(x)
        ident = FXTensor.identity_tensor(x)
        lhs = copy.composition(copy.tensor_product(ident))
        rhs = copy.composition(ident.tensor_product(copy))
        assert lhs == rhs
        assert lhs == FXTensor.copy_tensor(x, n=3)

    def test_counit(self, x):
        copy = FXTensor.copy_tensor(x)
        ident = FXTensor.identity_tensor(x)
        discard = FXTensor.exclamation(x)
        assert copy.composition(ident.tensor_product(discard)) == ident
        assert copy.composition(discard.tensor_product(ident)) == ident

    def test_copy_multiplicative(self, x):
        y = [2]
        lhs = FXTensor.copy_tensor(x + y)
        middle = FXTensor.identity_tensor(x).tensor_product(FXTensor.swap(x, y)).tensor_product(FXTensor.identity_tensor(y))
        rhs = FXTensor.copy_tensor(x).tensor_product(FXTensor.copy_tensor(y)).composition(middle)
        assert lhs == rhs


class TestMarkovProperties:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_discard_natural(self, shape):
        dom, cod = shape
        f = random_kernel(dom, cod, 4)
        assert f.composition(FXTensor.exclamation(cod)) == FXTensor.exclamation(dom)

    def test_kernels_closed_under_operations(self):
        f = random_kernel([2], [3], 0)
        g = random_kernel([3], [2, 2], 1)
        assert f.composition(g).is_markov()
        assert f.tensor_product(g).is_markov()

    def test_deterministic_commutes_with_copy(self):
        f = deterministic_kernel([2, 2], [3], 5)
        lhs = f.composition(FXTensor.copy_tensor([3]))
        rhs = FXTensor.copy_tensor([2, 2]).composition(f.tensor_product(f))
        assert lhs == rhs

    def test_random_kernel_does_not_commute_with_copy(self):
        f = random_kernel([2], [3], 6)
        lhs = f.composition(FXTensor.copy_tensor([3]))
        rhs = FXTensor.copy_tensor([2]).composition(f.tensor_product(f))
        assert lhs != rhs

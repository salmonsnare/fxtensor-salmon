"""Exchangeability, symmetrization, and Kolmogorov consistency.

``p: I → X^{⊗n}`` is exchangeable at position ``k`` when

``p ; (id^{⊗(k-1)} ⊗ swap(X, X) ⊗ id^{⊗(n-k-1)}) = p``.

A sequence is Kolmogorov consistent when

``p_n ; (id^{⊗(n-1)} ⊗ !) = p_{n-1}``.
"""
import warnings

import numpy as np
import pytest

from fxtensor_salmon import FXTensor
from test_axioms import random_kernel


def _coin(prob_heads=0.3):
    return FXTensor([[], [["H", "T"]]], data=np.array([prob_heads, 1.0 - prob_heads]))


def test_iid_is_exchangeable_and_kolmogorov_consistent():
    p = _coin(0.3)
    states = [FXTensor.iid(p, n) for n in range(1, 5)]
    assert states[0] == p
    assert states[2].labels == (None, [["H", "T"], ["H", "T"], ["H", "T"]])
    for state in states:
        assert state.is_exchangeable()
    assert FXTensor.is_kolmogorov_consistent(states)
    # n = 3 is the outer product, and it factors.
    expected = np.multiply.outer(np.multiply.outer(p.data, p.data), p.data)
    assert np.allclose(states[2].data, expected)
    assert states[1].is_independent(2)


def test_adjacent_swap_equation_transposes_wires():
    data = np.array([[0.1, 0.2], [0.3, 0.4]])
    p = FXTensor([[], [2, 2]], data=data)
    swap = FXTensor.swap([2], [2])
    assert np.allclose(p.composition(swap).data, data.T)
    assert not p.is_exchangeable()

    cube = np.arange(8, dtype=float).reshape(2, 2, 2)
    joint = FXTensor([[], [2, 2, 2]], data=cube)
    # k = 1: swap ⊗ id. k = 2: id ⊗ swap. No identity block on the short side.
    swap_first = FXTensor.swap([2], [2]).tensor_product(FXTensor.identity_tensor([2]))
    swap_last = FXTensor.identity_tensor([2]).tensor_product(FXTensor.swap([2], [2]))
    assert np.allclose(joint.composition(swap_first).data, np.transpose(cube, (1, 0, 2)))
    assert np.allclose(joint.composition(swap_last).data, np.transpose(cube, (0, 2, 1)))


def test_discard_last_wire_is_the_kolmogorov_equation():
    p1 = _coin(0.2)
    p2 = FXTensor.iid(p1, 2)
    discard = FXTensor.identity_tensor([2]).tensor_product(FXTensor.exclamation([2]))
    marginal = p2.composition(discard)
    assert np.allclose(marginal.data, p1.data)
    assert np.allclose(marginal.data, p2.marginalization(2).data)
    assert FXTensor.is_kolmogorov_consistent([p1, p2])


def test_mixture_of_iid_is_exchangeable_consistent_and_not_independent():
    bias = ["fair", "heads"]
    side = ["H", "T"]
    prior = FXTensor([[], [bias]], data=np.array([0.5, 0.5]))
    kernel = FXTensor(
        [[bias], [side]],
        data=np.array([
            [0.5, 0.5],
            [0.9, 0.1],
        ]),
    )
    states = [FXTensor.mixture_of_iid(prior, kernel, n) for n in range(1, 5)]
    assert states[2].labels == (None, [side, side, side])
    for state in states:
        assert state.is_exchangeable()
    assert FXTensor.is_kolmogorov_consistent(states)

    prior_data = np.array([0.25, 0.75])
    kernel_data = np.array([[0.2, 0.8], [0.6, 0.4]])
    numeric_prior = FXTensor([[], [2]], data=prior_data)
    numeric_kernel = FXTensor([[2], [2]], data=kernel_data)
    pair = FXTensor.mixture_of_iid(numeric_prior, numeric_kernel, 2)
    triple = FXTensor.mixture_of_iid(numeric_prior, numeric_kernel, 3)
    assert np.allclose(pair.data, np.einsum("t,tx,ty->xy", prior_data, kernel_data, kernel_data))
    assert np.allclose(
        triple.data,
        np.einsum("t,tx,ty,tz->xyz", prior_data, kernel_data, kernel_data, kernel_data),
    )
    marginal = pair.marginalization(2)
    assert not np.allclose(pair.data, np.multiply.outer(marginal.data, marginal.data))
    assert not pair.is_independent(2)

    point = FXTensor([[], [2]], data=np.array([0.0, 1.0]))
    fixed = FXTensor.mixture_of_iid(point, numeric_kernel, 3)
    pushed = FXTensor([[], [2]], data=kernel_data[1])
    assert fixed == FXTensor.iid(pushed, 3)
    assert fixed.is_independent(2)
    assert np.allclose(fixed.data, np.multiply.outer(np.multiply.outer(pushed.data, pushed.data), pushed.data))


def test_mixture_keeps_kernel_labels_when_prior_is_unlabeled():
    side = ["H", "T"]
    prior = FXTensor([[], [2]], data=np.array([0.4, 0.6]))
    kernel = FXTensor([[["a", "b"]], [side]], data=np.array([[0.2, 0.8], [0.7, 0.3]]))
    mixed = FXTensor.mixture_of_iid(prior, kernel, 2)
    assert mixed.labels == (None, [side, side])
    assert mixed.is_exchangeable()


def test_symmetrize_repairs_a_random_joint_and_is_idempotent():
    raw = random_kernel([], [3, 3, 3], 0)
    assert not raw.is_exchangeable()
    sym = raw.symmetrize()
    assert sym.is_exchangeable()
    assert sym.profile == raw.profile
    assert sym.symmetrize() == sym
    already = FXTensor.iid(_coin(0.4), 3)
    assert already.symmetrize() == already
    assert already.symmetrize().labels == already.labels


def test_inconsistent_marginal_is_not_kolmogorov_consistent():
    p1 = FXTensor([[], [2]], data=np.array([0.2, 0.8]))
    p2 = FXTensor([[], [2, 2]], data=np.array([[0.5, 0.0], [0.0, 0.5]]))
    assert np.allclose(p2.marginalization(2).data, [0.5, 0.5])
    assert not FXTensor.is_kolmogorov_consistent([p1, p2])
    assert FXTensor.is_kolmogorov_consistent([p1])
    assert FXTensor.is_kolmogorov_consistent([])


def test_mixed_factor_types_raise():
    mixed = FXTensor([[], [2, 3]], data=np.array([[0.2, 0.3, 0.1], [0.1, 0.2, 0.1]]))
    with pytest.raises(ValueError):
        mixed.is_exchangeable()
    with pytest.raises(ValueError):
        mixed.symmetrize()


def test_block_groups_a_composite_factor():
    rng = np.random.default_rng(1)
    data = rng.random((2, 3, 2, 3))
    data /= data.sum()
    raw = FXTensor([[], [2, 3, 2, 3]], data=data)
    with pytest.raises(ValueError):
        raw.is_exchangeable()
    sym = raw.symmetrize(block=[2, 3])
    assert sym.is_exchangeable(block=[2, 3])
    assert np.allclose(sym.data, np.transpose(sym.data, (2, 3, 0, 1)))
    assert sym.symmetrize(block=[2, 3]) == sym

    factor = FXTensor([[], [2, 3]], data=np.array([[0.1, 0.2, 0.05], [0.15, 0.3, 0.2]]))
    states = [FXTensor.iid(factor, n) for n in range(1, 4)]
    assert FXTensor.is_kolmogorov_consistent(states, block=[2, 3])
    with pytest.raises(ValueError):
        FXTensor.is_kolmogorov_consistent(states)


def test_non_state_and_bad_arguments_raise():
    kernel = FXTensor([[2], [2]], data=np.array([[0.2, 0.8], [0.4, 0.6]]))
    with pytest.raises(ValueError):
        kernel.is_exchangeable()
    with pytest.raises(ValueError):
        FXTensor.iid(kernel, 0)
    with pytest.raises(ValueError):
        FXTensor.mixture_of_iid(kernel, kernel, 2)
    prior = FXTensor([[], [3]], data=np.array([0.2, 0.3, 0.5]))
    with pytest.raises(ValueError):
        FXTensor.mixture_of_iid(prior, kernel, 2)
    with pytest.raises(ValueError):
        FXTensor.is_kolmogorov_consistent([kernel])
    p1 = _coin()
    p3 = FXTensor.iid(p1, 3)
    with pytest.raises(ValueError):
        FXTensor.is_kolmogorov_consistent([p1, p3])


def test_symmetrize_warns_above_eight_factors():
    state = FXTensor([[], [1] * 9], data=np.ones((1,) * 9))
    with pytest.warns(UserWarning, match="above 8"):
        result = state.symmetrize()
    assert result.is_exchangeable()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        FXTensor.iid(_coin(), 2).symmetrize()

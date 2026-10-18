"""Conditional independence and conditional products.

The codomain order used by the API is X ⊗ Y ⊗ Z. Fritz's string diagram
(arXiv:1908.07021, Def. 12.1) draws the same variables as X ⊗ Z ⊗ Y.
"""
import numpy as np
import pytest

from fxtensor_salmon import FXTensor


def _xz_marginal(joint, n_x, n_y):
    """Drop the middle ``n_y`` codomain factors of ``X ⊗ Y ⊗ Z``."""
    cod = joint.profile[1]
    x = cod[:n_x]
    y = cod[n_x : n_x + n_y]
    z = cod[n_x + n_y :]
    mover = FXTensor.identity_tensor(x).tensor_product(FXTensor.swap(y, z))
    return joint.composition(mover).marginalization(n_x + len(z) + 1)


def test_jointification_is_independent():
    px = FXTensor([[], [2]], data=np.array([0.3, 0.7]))
    py = FXTensor([[], [3]], data=np.array([0.2, 0.5, 0.3]))
    joint = px.jointification(py)
    assert joint.is_independent(2)
    assert joint.is_conditionally_independent(2, 3)


def test_mixture_of_iid_is_not_independent():
    """Equal mixture of two i.i.d. Bernoulli laws: exchangeable, not independent."""
    low = np.array([0.8, 0.2])
    high = np.array([0.2, 0.8])
    mixed = 0.5 * np.outer(low, low) + 0.5 * np.outer(high, high)
    joint = FXTensor([[], [2, 2]], data=mixed)
    assert np.allclose(joint.data, joint.data.T)
    assert not joint.is_independent(2)


def test_chain_is_conditionally_independent_but_not_independent():
    px = np.array([0.3, 0.7])
    f = np.array([[0.9, 0.1], [0.2, 0.8]])  # X → Z
    g = np.array([[0.8, 0.2], [0.1, 0.9]])  # Z → Y
    data = np.einsum("x,xz,zy->xyz", px, f, g)
    joint = FXTensor([[], [2, 2, 2]], data=data)
    assert joint.is_conditionally_independent(2, 3)
    xy = joint.marginalization(3)
    assert not xy.is_independent(2)


def test_conditional_product_matches_formula_and_marginals():
    p = np.array([[0.45, 0.05], [0.05, 0.45]])
    q = np.array([[0.40, 0.10], [0.10, 0.40]])
    pz = p.sum(axis=0)
    expected = (p / pz)[:, None, :] * (q / pz)[None, :, :] * pz
    p_xz = FXTensor([[], [2, 2]], data=p)
    q_yz = FXTensor([[], [2, 2]], data=q)
    got = FXTensor.conditional_product(p_xz, q_yz, 2)
    assert got.profile == [[], [2, 2, 2]]
    assert np.allclose(got.data, expected)
    assert got.is_conditionally_independent(2, 3)
    assert np.allclose(_xz_marginal(got, 1, 1).data, p)
    assert np.allclose(got.discard_prefix(2).data, q)


def test_conditional_product_multifactor_blocks():
    rng = np.random.default_rng(1)
    z = rng.random(2)
    z /= z.sum()
    px = rng.random((2, 2, 2))
    px /= px.sum(axis=(1, 2), keepdims=True)
    py = rng.random((2, 3))
    py /= py.sum(axis=1, keepdims=True)
    p_data = np.transpose(px, (1, 2, 0)) * z
    q_data = np.transpose(py, (1, 0)) * z
    expected = p_data[:, :, None, :] * (q_data / z)[None, None, :, :]
    p_xz = FXTensor([[], [2, 2, 2]], data=p_data)
    q_yz = FXTensor([[], [3, 2]], data=q_data)
    got = FXTensor.conditional_product(p_xz, q_yz, (3, 2))
    assert got.profile == [[], [2, 2, 3, 2]]
    assert np.allclose(got.data, expected)
    assert got.is_conditionally_independent(3, 4)
    assert np.allclose(_xz_marginal(got, 2, 1).data, p_data)
    assert np.allclose(got.discard_prefix(3).data, q_data)


def test_conditional_product_zero_slice_stays_finite():
    p = np.array([[0.5, 0.0], [0.5, 0.0]])
    q = np.array([[0.2, 0.0], [0.8, 0.0]])
    got = FXTensor.conditional_product(
        FXTensor([[], [2, 2]], data=p),
        FXTensor([[], [2, 2]], data=q),
        2,
    )
    assert np.all(np.isfinite(got.data))
    assert got.is_conditionally_independent(2, 3)
    assert np.allclose(got.data[:, :, 1], 0.0)


def test_conditional_product_of_kernels_sharing_a_domain():
    rng = np.random.default_rng(2)
    pz = rng.random((2, 3))
    pz /= pz.sum(axis=1, keepdims=True)
    px = rng.random((2, 3, 2))
    px /= px.sum(axis=2, keepdims=True)
    py = rng.random((2, 3, 4))
    py /= py.sum(axis=2, keepdims=True)
    p_data = np.transpose(px, (0, 2, 1)) * pz[:, None, :]
    q_data = np.transpose(py, (0, 2, 1)) * pz[:, None, :]
    expected = (
        np.transpose(px, (0, 2, 1))[:, :, None, :]
        * np.transpose(py, (0, 2, 1))[:, None, :, :]
        * pz[:, None, None, :]
    )
    context = ["Home", "Away"]
    weather = ["Dry", "Wet"]
    umbrella = ["Leave", "Take", "Either", "Neither"]
    forecast = ["Sun", "Cloud", "Rain"]
    p_xz = FXTensor([[context], [weather, forecast]], data=p_data)
    q_yz = FXTensor([[context], [umbrella, forecast]], data=q_data)
    got = FXTensor.conditional_product(p_xz, q_yz, 2)
    assert got.labels == ([context], [weather, umbrella, forecast])
    assert np.allclose(got.data, expected)
    assert got.is_conditionally_independent(2, 3)
    assert np.allclose(_xz_marginal(got, 1, 1).data, p_data)
    assert np.allclose(got.discard_prefix(2).data, q_data)


def test_kernel_independent_given_input_when_z_is_trivial():
    hx = np.array([[0.2, 0.8], [0.5, 0.5]])
    hy = np.array([[0.3, 0.7], [0.6, 0.4]])
    joint = hx[:, :, None] * hy[:, None, :]
    kernel = FXTensor([[2], [2, 2]], data=joint)
    assert kernel.is_independent(2)
    coupled = np.array([[[0.4, 0.1], [0.1, 0.4]], [[0.5, 0.0], [0.0, 0.5]]])
    assert not FXTensor([[2], [2, 2]], data=coupled).is_independent(2)


def test_generic_joint_is_not_conditionally_independent():
    rng = np.random.default_rng(0)
    data = rng.random((2, 2, 2))
    data /= data.sum()
    joint = FXTensor([[], [2, 2, 2]], data=data)
    assert not joint.is_conditionally_independent(2, 3)


def test_labeled_conditional_product_keeps_labels():
    weather = ["Dry", "Wet"]
    umbrella = ["Leave", "Take"]
    forecast = ["Sunny", "Rainy"]
    p_wf = FXTensor(
        [[], [weather, forecast]],
        data=np.array([[0.45, 0.05], [0.05, 0.45]]),
    )
    p_uf = FXTensor(
        [[], [umbrella, forecast]],
        data=np.array([[0.40, 0.10], [0.10, 0.40]]),
    )
    joint = FXTensor.conditional_product(p_wf, p_uf, 2)
    assert joint.labels == (None, [weather, umbrella, forecast])
    marginal = joint.marginalization(3)
    assert marginal.labels == (None, [weather, umbrella])
    assert not marginal.is_independent(2)


def test_atol_rtol_are_exposed():
    p = np.array([[0.45, 0.05], [0.05, 0.45]])
    q = np.array([[0.40, 0.10], [0.10, 0.40]])
    base = FXTensor.conditional_product(
        FXTensor([[], [2, 2]], data=p),
        FXTensor([[], [2, 2]], data=q),
        2,
    )
    noisy = base.copy()
    noisy.data = noisy.data + 1e-6
    assert not noisy.is_conditionally_independent(2, 3, atol=1e-9, rtol=0.0)
    assert noisy.is_conditionally_independent(2, 3, atol=1e-4, rtol=0.0)


def test_split_out_of_range_raises():
    joint = FXTensor([[], [2, 2]], data=np.array([[0.2, 0.3], [0.1, 0.4]]))
    with pytest.raises(ValueError):
        joint.is_conditionally_independent(0, 2)
    with pytest.raises(ValueError):
        joint.is_conditionally_independent(2, 1)
    with pytest.raises(ValueError):
        joint.is_conditionally_independent(1, 4)
    with pytest.raises(ValueError):
        joint.is_conditionally_independent(True, 2)
    with pytest.raises(ValueError):
        joint.is_independent(0)
    with pytest.raises(ValueError):
        joint.is_independent(4)
    with pytest.raises(ValueError):
        FXTensor.conditional_product(joint, joint, 0)
    with pytest.raises(ValueError):
        FXTensor.conditional_product(joint, joint, 4)
    with pytest.raises(ValueError):
        FXTensor.conditional_product(joint, joint, (2, 2, 2))


def test_mismatched_z_raises():
    p = FXTensor([[], [2, 2]], data=np.array([[0.4, 0.1], [0.4, 0.1]]))
    other_size = FXTensor([[], [2, 3]], data=np.ones((2, 3)) / 6.0)
    other_marginal = FXTensor([[], [2, 2]], data=np.array([[0.1, 0.4], [0.1, 0.4]]))
    other_domain = FXTensor([[2], [2, 2]], data=np.ones((2, 2, 2)) / 4.0)
    with pytest.raises(ValueError, match="mismatched Z"):
        FXTensor.conditional_product(p, other_size, 2)
    with pytest.raises(ValueError, match="mismatched Z"):
        FXTensor.conditional_product(p, other_marginal, 2)
    with pytest.raises(ValueError, match="domain"):
        FXTensor.conditional_product(p, other_domain, 2)

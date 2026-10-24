"""Axis names: permute, marginal, marginalize, and observe.

Names are variable names on wires, not the labels of the points on a wire.
``names=None`` leaves a tensor unnamed.
"""
import numpy as np
import pytest

from fxtensor_salmon import FXTensor


def _state():
    """State ``I → A ⊗ B ⊗ C`` with names ``a, b, c`` and distinct labels."""
    a = ["p", "q"]
    b = ["x", "y", "z"]
    c = ["u", "v", "w", "t"]
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    data = data / data.sum()
    tensor = FXTensor([[], [a, b, c]], data=data, names=([], ["a", "b", "c"]))
    return tensor, data, a, b, c


def test_marginal_matches_sum_and_wiring_and_reorders_labels():
    state, data, a, b, c = _state()
    got = state.marginal("b")
    assert np.allclose(got.data, data.sum(axis=(0, 2)))
    assert got.labels == (None, [b])
    assert got.names == (None, ["b"])

    wire = np.zeros((2, 3, 4, 3))
    for i, j, k in np.ndindex(2, 3, 4):
        wire[i, j, k, j] = 1.0
    composed = state.composition(FXTensor([[2, 3, 4], [3]], data=wire))
    assert composed.profile == got.profile
    assert np.allclose(composed.data, got.data)

    both = state.marginal("c", "a")
    assert both.names == (None, ["c", "a"])
    assert both.labels == (None, [c, a])
    assert np.allclose(both.data, data.sum(axis=1).transpose(1, 0))
    assert state.marginalize(drop=["a", "c"]) == state.marginal("b")


def test_permute_matches_transpose_and_moves_labels():
    state, data, a, b, c = _state()
    got = state.permute(["c", "a", "b"])
    assert np.allclose(got.data, data.transpose(2, 0, 1))
    assert got.labels == (None, [c, a, b])
    assert got.names == (None, ["c", "a", "b"])
    assert got.axis("c") == 0
    assert state.axis("b") == 1


def test_observe_matches_conditionalization_and_marginal():
    state, data, _a, b, _c = _state()
    post, probability = state.observe(b="x")
    sl = data[:, 0, :]
    assert np.allclose(post.data, sl / sl.sum())
    assert post.names == (None, ["a", "c"])
    assert np.isclose(probability, state.marginal("b").data[0])

    moved = state.permute(["b", "a", "c"])
    cond = moved.conditionalization(2)
    idx = cond.get_label_index(0, "x")
    expected = FXTensor(
        [[], [state.labels[1][0], state.labels[1][2]]],
        data=cond.data[idx],
        names=([], ["a", "c"]),
    )
    assert post == expected
    assert np.isclose(probability, float(state.marginal("b").data[b.index("x")]))


def test_observe_several_names_matches_sequential_observe():
    state, _data, _a, _b, _c = _state()
    post, probability = state.observe(a="p", c="u")
    first, p_a = state.observe(a="p")
    second, p_c = first.observe(c="u")
    assert post == second
    assert np.isclose(probability, p_a * p_c)
    assert post.names == (None, ["b"])


def test_zero_evidence_raises():
    _state_tensor, data, a, b, c = _state()
    data = data.copy()
    data[:, 1, :] = 0
    data = data / data.sum()
    tensor = FXTensor([[], [a, b, c]], data=data, names=([], ["a", "b", "c"]))
    with pytest.raises(ValueError):
        tensor.observe(b="y")


def test_same_labels_are_distinguished_by_name():
    level = ["low", "high"]
    raw = np.array([[0.1, 0.2], [0.3, 0.4]])
    twin = FXTensor([[], [level, level]], data=raw, names=([], ["E", "F"]))
    assert not np.allclose(twin.marginal("E").data, twin.marginal("F").data)
    assert twin.marginal("E").names == (None, ["E"])
    assert twin.marginal("F").names == (None, ["F"])


def test_bad_names_raise():
    state, _data, _a, _b, _c = _state()
    with pytest.raises(ValueError):
        FXTensor([[], [2, 2]], data=np.ones((2, 2)), names=([], ["a", "a"]))
    with pytest.raises(ValueError):
        FXTensor(
            [[["u", "v"]], [["x", "y"]]],
            data=np.eye(2),
            names=(["A"], ["A"]),
        )
    with pytest.raises(ValueError):
        state.marginal("missing")
    with pytest.raises(ValueError):
        state.marginalize(drop=["missing"])
    with pytest.raises(ValueError):
        state.permute(["a", "b"])
    with pytest.raises(ValueError):
        state.observe(b="missing")
    with pytest.raises(ValueError):
        state.observe(missing="x")
    with pytest.raises(ValueError):
        FXTensor([[2], [2]], data=np.eye(2), names=(["A"], ["B"])).observe(B=0)


def test_rename_and_kernel_axis_numbers():
    state, _data, _a, _b, _c = _state()
    renamed = state.rename(a="alpha", c="gamma")
    assert renamed.names == (None, ["alpha", "b", "gamma"])
    assert np.allclose(renamed.marginal("b").data, state.marginal("b").data)
    with pytest.raises(ValueError):
        state.rename(a="b")

    kernel = FXTensor(
        [[["u", "v"]], [["p", "q"], ["x", "y"]]],
        data=np.arange(8, dtype=float).reshape(2, 2, 2),
        names=(["D"], ["A", "B"]),
    )
    assert kernel.axis("D") == 0
    assert kernel.axis("A") == 1
    assert kernel.axis("B") == 2
    swapped = kernel.permute(["B", "A"])
    assert swapped.names == (["D"], ["B", "A"])
    assert swapped.labels == ([["u", "v"]], [["x", "y"], ["p", "q"]])
    assert kernel.marginal("B").names == (["D"], ["B"])
    assert kernel.marginal("B").profile == [[2], [2]]
    assert np.allclose(kernel.marginal("B").data, kernel.data.sum(axis=1))


def test_name_propagation_and_json():
    level = ["low", "high"]
    prior = FXTensor([[], [level]], data=[0.25, 0.75], names=([], ["A"]))
    kernel = FXTensor(
        [[level], [level]],
        data=[[0.8, 0.2], [0.4, 0.6]],
        names=(["A"], ["C"]),
    )
    pushed = prior.composition(kernel)
    assert pushed.names == (None, ["C"])
    mismatch = FXTensor(
        [[level], [level]],
        data=[[0.8, 0.2], [0.4, 0.6]],
        names=(["Z"], ["C"]),
    )
    with pytest.raises(ValueError):
        prior.composition(mismatch)

    other = FXTensor([[], [level]], data=[0.3, 0.7], names=([], ["B"]))
    assert prior.tensor_product(other).names == (None, ["A", "B"])
    with pytest.raises(ValueError):
        prior.tensor_product(prior)

    assert kernel.bayesian_inversion(prior).names == (["C"], ["A"])
    joint = prior.tensor_product(other)
    assert joint.conditionalization(2).names == (["A"], ["B"])
    wider = FXTensor(
        [[["u", "v"]], [level, level]],
        data=np.arange(8, dtype=float).reshape(2, 2, 2),
        names=(["D"], ["A", "B"]),
    )
    assert wider.conditional(2).names == (["D", "A"], ["B"])
    assert wider.support().names == wider.names
    assert wider.marginalization(2).names == (["D"], ["A"])
    assert wider.discard_prefix(2).names == (["D"], ["B"])

    blob = joint.to_json()
    assert blob["names"] == [[], ["A", "B"]]
    assert FXTensor.from_json(blob) == joint
    bare = {"profile": [[], [2]], "data": [0.25, 0.75], "dtype": "float64"}
    loaded = FXTensor.from_json(bare)
    assert loaded.names == (None, None)
    assert "names" not in loaded.to_json()


def test_chain_observe_matches_bayesian_inversion():
    level = ["low", "high"]
    prior_data = np.array([0.25, 0.75])
    given_a = np.array([[0.8, 0.2], [0.4, 0.6]])
    given_c = np.array([[0.9, 0.1], [0.3, 0.7]])
    joint_data = prior_data[:, None, None] * given_a[:, :, None] * given_c[None, :, :]
    joint = FXTensor(
        [[], [level, level, level]],
        data=joint_data,
        names=([], ["A", "C", "D"]),
    )
    prior = FXTensor([[], [level]], data=prior_data, names=([], ["A"]))
    kernel = FXTensor([[level], [level]], data=given_a, names=(["A"], ["C"]))
    inversion = kernel.bayesian_inversion(prior)
    post, evidence = joint.observe(C="high")
    assert post.names == (None, ["A", "D"])
    assert np.allclose(post.marginal("A").data, inversion.data[1])
    assert np.isclose(evidence, joint.marginal("C").data[1])


def test_unnamed_tensor_has_no_names():
    left = FXTensor([[2], [3]], data=np.ones((2, 3)) / 3)
    right = FXTensor([[3], [2]], data=np.ones((3, 2)) / 2)
    assert left.names == (None, None)
    got = left.composition(right)
    assert got.names == (None, None)
    assert got.profile == [[2], [2]]
    assert "names" not in left.to_json()


def test_integer_evidence_without_labels():
    data = np.array([[0.1, 0.2], [0.3, 0.4]])
    tensor = FXTensor([[], [2, 2]], data=data, names=([], ["a", "b"]))
    post, probability = tensor.observe(b=0)
    assert np.allclose(post.data, data[:, 0] / data[:, 0].sum())
    assert np.isclose(probability, data[:, 0].sum())
    assert post.names == (None, ["a"])

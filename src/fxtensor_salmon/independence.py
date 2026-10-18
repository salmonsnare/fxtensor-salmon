"""Conditional independence for finite Markov kernels.

Fritz's copy–discard equation (arXiv:1908.07021, Definitions 12.1 and 12.16)
and the conditional product (Definition 12.8). Codomains are ordered
``X ⊗ Y ⊗ Z``. The string diagram draws the same variables as ``X ⊗ Z ⊗ Y``.
"""

from __future__ import annotations

import numpy as np


def _tolerances(cls, atol, rtol):
    if atol is None:
        atol = cls._ATOL
    if rtol is None:
        rtol = cls._RTOL
    return atol, rtol


def _is_index(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_ci_splits(split_x, split_y, n_factors):
    if not _is_index(split_x) or not _is_index(split_y):
        raise ValueError("split index is out of range")
    if not (1 <= split_x <= split_y <= n_factors + 1):
        raise ValueError("split index is out of range")


def _parse_on(on, n_p, n_q):
    """Return 1-based starts of the trailing ``Z`` block in each codomain."""
    if isinstance(on, tuple):
        if len(on) != 2:
            raise ValueError("on must be a split index or a pair of split indices")
        on_p, on_q = on
    else:
        on_p = on_q = on
    for value, n_factors in ((on_p, n_p), (on_q, n_q)):
        if not _is_index(value) or not (1 <= value <= n_factors + 1):
            raise ValueError("split index is out of range")
    return on_p, on_q


def _tensor_all(cls, factors):
    acc = factors[0]
    for factor in factors[1:]:
        acc = acc.tensor_product(factor)
    return acc


def _conditional_given_suffix(tensor, split):
    """Kernel ``A ⊗ C → B`` from ``tensor: A → B ⊗ C``.

    ``split`` is the 1-based first factor of ``C``. An empty ``B`` yields
    the discard of ``A ⊗ C``. An empty ``C`` yields the row-normalized
    marginal on ``B``.
    """
    prefix = list(tensor._profile[1][: split - 1])
    suffix = list(tensor._profile[1][split - 1 :])
    domain = list(tensor._profile[0])
    cls = type(tensor)
    if not prefix:
        dims = domain + suffix
        if not dims:
            return cls([[], []], data=np.array(1.0))
        return cls.exclamation(dims)
    if not suffix:
        return tensor.conditional(1)
    swapped = tensor.composition(cls.swap(prefix, suffix))
    return swapped.conditional(len(suffix) + 1)


def _marginal_on_suffix(tensor, split):
    """Marginal on the codomain suffix that starts at ``split`` (1-based)."""
    prefix = list(tensor._profile[1][: split - 1])
    suffix = list(tensor._profile[1][split - 1 :])
    if not prefix:
        return tensor
    cls = type(tensor)
    swapped = tensor.composition(cls.swap(prefix, suffix))
    return swapped.marginalization(len(suffix) + 1)


def _marginal_keeping(tensor, start, end):
    """Drop codomain factors ``[start, end)`` (1-based) and keep the rest."""
    if start == end:
        return tensor
    cod = tensor._profile[1]
    kept_prefix = list(cod[: start - 1])
    dropped = list(cod[start - 1 : end - 1])
    kept_suffix = list(cod[end - 1 :])
    cls = type(tensor)
    mover = cls.identity_tensor(kept_prefix).tensor_product(cls.swap(dropped, kept_suffix))
    moved = tensor.composition(mover)
    return moved.marginalization(len(kept_prefix) + len(kept_suffix) + 1)


def _merged_labels(p_xz, q_yz, on_p, on_q):
    if p_xz._labels is None or q_yz._labels is None:
        return None
    if p_xz._labels[0] != q_yz._labels[0]:
        return None
    z_p = p_xz._labels[1][on_p - 1 :]
    z_q = q_yz._labels[1][on_q - 1 :]
    if z_p != z_q:
        return None
    x_labels = p_xz._labels[1][: on_p - 1]
    y_labels = q_yz._labels[1][: on_q - 1]
    return (list(p_xz._labels[0]), list(x_labels) + list(y_labels) + list(z_p))


def _empty_z_product(cls, h, k, g):
    """``mass ; (h ⊗ k)`` when ``Z`` is the unit object."""
    a_dims = list(g._profile[0])
    if a_dims:
        paired = cls.copy_tensor(a_dims, n=2).composition(h.tensor_product(k))
        cod_rank = len(paired._profile[1])
        weights = np.reshape(np.asarray(g.data, dtype=float), a_dims + [1] * cod_rank)
        return paired._spawn(
            paired._profile[0],
            paired._profile[1],
            None,
            paired.data * weights,
        )
    return g.composition(h.tensor_product(k))


def _diagram_product(cls, h, k, g):
    """``p_Z ; copy_Z ; (p(x|z) ⊗ id_Z ⊗ q(y|z))``, then ``Z`` past ``Y``.

    The diagram's codomain is ``X ⊗ Z ⊗ Y``. The returned codomain is
    ``X ⊗ Y ⊗ Z``. For a kernel the input ``A`` is copied so both
    conditionals can read ``(a, z)``.
    """
    a_dims = list(g._profile[0])
    x_dims = list(h._profile[1])
    y_dims = list(k._profile[1])
    z_dims = list(g._profile[1])
    az = cls.copy_tensor(a_dims, n=2).composition(
        cls.identity_tensor(a_dims).tensor_product(g)
    )
    expanded = az.composition(
        cls.copy_tensor(a_dims, n=2).tensor_product(cls.copy_tensor(z_dims, n=3))
    )
    # (A, A, Z, Z, Z) → (A, Z, Z, A, Z), i.e. domain of h ⊗ id_Z ⊗ k.
    to_hxzk = _tensor_all(
        cls,
        [
            cls.identity_tensor(a_dims),
            cls.swap(a_dims, z_dims),
            cls.identity_tensor(z_dims),
            cls.identity_tensor(z_dims),
        ],
    )
    to_h_id_k = _tensor_all(
        cls,
        [
            cls.identity_tensor(a_dims),
            cls.identity_tensor(z_dims),
            cls.swap(a_dims, z_dims),
            cls.identity_tensor(z_dims),
        ],
    )
    aligned = expanded.composition(to_hxzk).composition(to_h_id_k)
    diagram = aligned.composition(
        _tensor_all(cls, [h, cls.identity_tensor(z_dims), k])
    )
    return diagram.composition(
        cls.identity_tensor(x_dims).tensor_product(cls.swap(z_dims, y_dims))
    )


def _assemble(p_xz, q_yz, on_p, on_q):
    h = _conditional_given_suffix(p_xz, on_p)
    k = _conditional_given_suffix(q_yz, on_q)
    g = _marginal_on_suffix(p_xz, on_p)
    cls = type(p_xz)
    if g._profile[1]:
        built = _diagram_product(cls, h, k, g)
    else:
        built = _empty_z_product(cls, h, k, g)
    labels = _merged_labels(p_xz, q_yz, on_p, on_q)
    if labels is None:
        return built
    return built._spawn(built._profile[0], built._profile[1], labels, built.data)


class IndependenceMixin:
    """Mixin that adds conditional independence to ``FXTensor``.

    Codomains are ordered ``X ⊗ Y ⊗ Z``. The string diagram draws the same
    variables as ``X ⊗ Z ⊗ Y``.
    """

    def is_conditionally_independent(self, split_x, split_y, atol=None, rtol=None):
        """Return whether ``X ⊥ Y | Z`` for codomain order ``X ⊗ Y ⊗ Z``.

        ``split_x`` and ``split_y`` are 1-based, like ``conditional``.
        Factors ``1 .. split_x - 1`` are ``X``, factors ``split_x .. split_y - 1``
        are ``Y``, and the remaining factors are ``Z``. Any of the three
        blocks may be empty. A state ``I → X ⊗ Y ⊗ Z`` and a kernel
        ``A → X ⊗ Y ⊗ Z`` use the same splits.

        The predicate holds when ``self`` equals the conditional product of
        its ``X ⊗ Z`` and ``Y ⊗ Z`` marginals: the finite form of Fritz's
        copy–discard equation (arXiv:1908.07021, Def. 12.1 for states and
        Def. 12.16 for kernels). Values are compared with ``numpy.allclose``.
        ``atol`` and ``rtol`` default to ``FXTensor._ATOL`` and
        ``FXTensor._RTOL``.

        Raises
        ------
        ValueError
            If a split is not a 1-based index in range.
        """
        n_factors = len(self._profile[1])
        _validate_ci_splits(split_x, split_y, n_factors)
        atol, rtol = _tolerances(type(self), atol, rtol)
        p_xz = _marginal_keeping(self, split_x, split_y)
        p_yz = _marginal_keeping(self, 1, split_x)
        on_p = split_x
        on_q = split_y - split_x + 1
        rebuilt = _assemble(p_xz, p_yz, on_p, on_q)
        if self.data.shape != rebuilt.data.shape:
            return False
        return bool(np.allclose(self.data, rebuilt.data, rtol=rtol, atol=atol))

    def is_independent(self, split, atol=None, rtol=None):
        """Return whether ``X ⊥ Y`` when the codomain is ``X ⊗ Y``.

        ``split`` is the 1-based first factor of ``Y``. This is
        :meth:`is_conditionally_independent` with an empty ``Z``.
        ``atol`` and ``rtol`` are forwarded.

        Raises
        ------
        ValueError
            If ``split`` is not a 1-based index in range.
        """
        n_factors = len(self._profile[1])
        return self.is_conditionally_independent(split, n_factors + 1, atol=atol, rtol=rtol)

    @classmethod
    def conditional_product(cls, p_xz, q_yz, on, atol=None, rtol=None):
        """Return the conditional product, with codomain ``X ⊗ Y ⊗ Z``.

        ``p_xz`` has codomain ``X ⊗ Z`` and ``q_yz`` has codomain ``Y ⊗ Z``.
        ``on`` is the 1-based index of the first ``Z`` factor in both
        codomains, or a pair ``(on_p, on_q)`` when ``X`` and ``Y`` have
        different numbers of factors. Both arguments are states, or both
        are kernels with the same domain.

        The value is ``p_Z ; copy_Z ; (p(x|z) ⊗ id_Z ⊗ q(y|z))`` with ``Z``
        then moved past ``Y``. That diagram has wires ``X ⊗ Z ⊗ Y``; the
        tensor returned here is ``X ⊗ Y ⊗ Z``, which satisfies
        ``X ⊥ Y | Z``. ``p_Z`` is the ``Z`` marginal of ``p_xz``. The two
        ``Z`` marginals must agree within ``atol`` and ``rtol`` (defaults:
        ``FXTensor._ATOL`` and ``FXTensor._RTOL``).

        Raises
        ------
        ValueError
            If ``on`` is out of range, the domains differ, the ``Z``
            factor sizes differ, or the ``Z`` marginals disagree.
        """
        if not isinstance(p_xz, cls) or not isinstance(q_yz, cls):
            raise ValueError("conditional_product expects FXTensor arguments")
        if list(p_xz._profile[0]) != list(q_yz._profile[0]):
            raise ValueError("conditional_product arguments must share a domain")
        on_p, on_q = _parse_on(on, len(p_xz._profile[1]), len(q_yz._profile[1]))
        z_p = list(p_xz._profile[1][on_p - 1 :])
        z_q = list(q_yz._profile[1][on_q - 1 :])
        if z_p != z_q:
            raise ValueError("mismatched Z")
        atol, rtol = _tolerances(cls, atol, rtol)
        g_p = _marginal_on_suffix(p_xz, on_p)
        g_q = _marginal_on_suffix(q_yz, on_q)
        if g_p.data.shape != g_q.data.shape or not np.allclose(
            g_p.data, g_q.data, rtol=rtol, atol=atol
        ):
            raise ValueError("mismatched Z")
        return _assemble(p_xz, q_yz, on_p, on_q)

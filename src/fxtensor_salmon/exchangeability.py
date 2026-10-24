"""Exchangeability and Kolmogorov consistency for finite states.

A state ``p: I → X^{⊗n}`` is exchangeable at position ``k`` (1-based) when

``p ; (id^{⊗(k-1)} ⊗ swap(X, X) ⊗ id^{⊗(n-k-1)}) = p``.

Swapping the ``k``-th and ``(k+1)``-th wires leaves ``p`` unchanged.
Those wires use ``(k-1) + 2 + (n-k-1) = n`` factors.

A sequence ``p_n: I → X^{⊗n}`` is Kolmogorov consistent when

``p_n ; (id^{⊗(n-1)} ⊗ !) = p_{n-1}``.

Discarding the ``n``-th wire of ``p_n`` yields ``p_{n-1}``.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np


_SYMMETRIZE_WARN_ABOVE = 8


def _tolerances(cls, atol, rtol):
    if atol is None:
        atol = cls._ATOL
    if rtol is None:
        rtol = cls._RTOL
    return atol, rtol


def _parse_block(block):
    """Return the dimension list of one factor ``X``."""
    if isinstance(block, tuple):
        block = list(block)
    if not isinstance(block, list) or not block:
        raise ValueError("block must be a non-empty list of positive dimensions")
    dims = []
    for axis in block:
        if isinstance(axis, bool) or not isinstance(axis, int) or axis <= 0:
            raise ValueError("block must be a non-empty list of positive dimensions")
        dims.append(axis)
    return dims


def _factors(codomain, block):
    """Split a codomain into ``n`` copies of ``X``.

    ``block is None`` means each axis is one factor (the default of one
    axis). Every axis must then have the same size. Otherwise ``block`` is
    the dimension list of ``X`` and must tile the codomain.
    """
    codomain = list(codomain)
    if block is None:
        if not codomain:
            return [], 0
        size = codomain[0]
        if any(axis != size for axis in codomain):
            raise ValueError(
                "codomain factors are not the same type X; pass block to group axes"
            )
        return [size], len(codomain)
    dims = _parse_block(block)
    width = len(dims)
    if len(codomain) % width != 0:
        raise ValueError("codomain length is not a multiple of the block")
    count = len(codomain) // width
    for index in range(count):
        piece = codomain[index * width:(index + 1) * width]
        if piece != dims:
            raise ValueError("codomain factors are not the same type X")
    return dims, count


def _require_state(tensor, method):
    if tensor._profile[0]:
        raise ValueError(f"{method} expects a state I → X^{{⊗n}}")


def _tensor_all(cls, factors):
    acc = factors[0]
    for factor in factors[1:]:
        acc = acc.tensor_product(factor)
    return acc


def _adjacent_swap_morphism(cls, x_dims, k, n):
    """``id^{⊗(k-1)} ⊗ swap(X, X) ⊗ id^{⊗(n-k-1)}``.

    ``k`` is 1-based. The three blocks use ``(k-1) + 2 + (n-k-1) = n``
    factors of ``X``.
    """
    if not (1 <= k <= n - 1):
        raise ValueError("swap position k is out of range")
    factors = []
    left = k - 1
    right = n - k - 1
    if left:
        factors.append(cls.identity_tensor(list(x_dims) * left))
    factors.append(cls.swap(list(x_dims), list(x_dims)))
    if right:
        factors.append(cls.identity_tensor(list(x_dims) * right))
    return _tensor_all(cls, factors)


def _discard_last_morphism(cls, x_dims, n):
    """``id^{⊗(n-1)} ⊗ !`` on ``X^{⊗n}``, discarding the ``n``-th wire."""
    if n < 1:
        raise ValueError("n must be positive")
    factors = []
    if n - 1:
        factors.append(cls.identity_tensor(list(x_dims) * (n - 1)))
    factors.append(cls.exclamation(list(x_dims)))
    return _tensor_all(cls, factors)


def _same_data(left, right, atol, rtol):
    if left.shape != right.shape:
        return False
    return bool(np.allclose(left, right, rtol=rtol, atol=atol))


def _permutation_axes(perm, width):
    order = []
    for block_index in perm:
        start = block_index * width
        order.extend(range(start, start + width))
    return order


class ExchangeabilityMixin:
    """Mixin that adds exchangeability and Kolmogorov consistency.

    States are morphisms ``I → X^{⊗n}``. ``X`` is one codomain axis unless
    ``block`` gives the dimension list of a multi-axis factor.
    """

    def is_exchangeable(self, block=None, atol=None, rtol=None):
        """Return whether this state is invariant under every adjacent swap.

        The state ``p: I → X^{⊗n}`` is exchangeable at position ``k``
        (1-based) when

        ``p ; (id^{⊗(k-1)} ⊗ swap(X, X) ⊗ id^{⊗(n-k-1)}) = p``.

        Swapping the ``k``-th and ``(k+1)``-th wires leaves ``p`` unchanged.
        The identity wires and the two swapped wires use
        ``(k-1) + 2 + (n-k-1) = n`` factors. Adjacent transpositions
        generate the symmetric group, so this checks ``k = 1 .. n-1``.
        ``n < 2`` is vacuously true.

        ``block`` is the dimension list of one factor ``X``. The default,
        ``None``, takes each codomain axis as one factor; those axes must
        share a size. ``swap`` carries no labels, so the comparison is on
        ``data`` via ``numpy.allclose``. ``atol`` and ``rtol`` default to
        ``FXTensor._ATOL`` and ``FXTensor._RTOL``.

        Raises
        ------
        ValueError
            If ``self`` is not a state, or the codomain is not ``X^{⊗n}``.
        """
        _require_state(self, "is_exchangeable")
        x_dims, count = _factors(self._profile[1], block)
        if count < 2:
            return True
        atol, rtol = _tolerances(type(self), atol, rtol)
        cls = type(self)
        for k in range(1, count):
            swapped = self.composition(_adjacent_swap_morphism(cls, x_dims, k, count))
            if not _same_data(swapped.data, self.data, atol, rtol):
                return False
        return True

    def symmetrize(self, block=None):
        """Return the average of this state over every permutation of ``X``.

        The result has the same profile and labels and satisfies
        :meth:`is_exchangeable` for the same ``block``. Averaging again
        changes nothing, and an exchangeable state is left unchanged.
        ``n > 8`` warns: the average runs through ``n!`` permutations.

        Raises
        ------
        ValueError
            If ``self`` is not a state, or the codomain is not ``X^{⊗n}``.
        """
        _require_state(self, "symmetrize")
        x_dims, count = _factors(self._profile[1], block)
        if count > _SYMMETRIZE_WARN_ABOVE:
            warnings.warn(
                f"symmetrize averages {count}! permutations; n={count} is above 8",
                stacklevel=2,
            )
        if count < 2:
            return self.copy()
        width = len(x_dims)
        total = np.zeros(self.data.shape, dtype=float)
        n_perms = 0
        for perm in itertools.permutations(range(count)):
            total += np.transpose(self.data, _permutation_axes(perm, width))
            n_perms += 1
        averaged = total / n_perms
        return self._spawn(self._profile[0], self._profile[1], self._labels, averaged, self._names)

    @classmethod
    def is_kolmogorov_consistent(cls, states, block=None, atol=None, rtol=None):
        """Return whether discarding the last factor reproduces the previous state.

        ``states`` is the list ``[p_1, p_2, ..., p_N]`` with
        ``p_n: I → X^{⊗n}`` (so ``states[0]`` is ``p_1``). The sequence is
        Kolmogorov consistent when, for every ``n ≥ 2``,

        ``p_n ; (id^{⊗(n-1)} ⊗ !) = p_{n-1}``.

        ``!`` is :meth:`exclamation` on the last factor: discarding the
        ``n``-th wire of ``p_n`` yields ``p_{n-1}``. A list of length 0 or
        1 is consistent. ``block`` selects ``X`` as in
        :meth:`is_exchangeable`; the default is one axis, and
        ``states[n-1]`` must then have ``n`` axes of that size. Values are
        compared on ``data`` with ``numpy.allclose``. ``atol`` and ``rtol``
        default to ``FXTensor._ATOL`` and ``FXTensor._RTOL``.

        Raises
        ------
        ValueError
            If an entry is not a state ``I → X^{⊗n}`` for that position, or
            the factors are not a common type ``X``.
        """
        try:
            sequence = list(states)
        except TypeError as exc:
            raise ValueError("states must be a sequence of FXTensor states") from exc
        for state in sequence:
            if not isinstance(state, cls):
                raise ValueError("states must be a sequence of FXTensor states")
            _require_state(state, "is_kolmogorov_consistent")
        if not sequence:
            return True
        parsed = [_factors(state._profile[1], block) for state in sequence]
        x_dims = parsed[0][0]
        if not x_dims:
            raise ValueError("codomain must be X^{⊗n} for a non-empty X")
        for index, (dims, count) in enumerate(parsed):
            n = index + 1
            if dims != x_dims or count != n:
                raise ValueError(f"states[{index}] must be a state I → X^{{⊗{n}}}")
        if len(sequence) < 2:
            return True
        atol, rtol = _tolerances(cls, atol, rtol)
        for n in range(2, len(sequence) + 1):
            discarded = sequence[n - 1].composition(
                _discard_last_morphism(cls, x_dims, n)
            )
            if not _same_data(discarded.data, sequence[n - 2].data, atol, rtol):
                return False
        return True

    @classmethod
    def iid(cls, p, n):
        """Return the tensor product of ``n`` copies of ``p``.

        Labels are taken from ``p``. For a state ``I → X`` the result is
        ``I → X^{⊗n}``. ``n=1`` returns a copy of ``p``.

        Raises
        ------
        ValueError
            If ``p`` is not an ``FXTensor``, or ``n`` is not a positive integer.
        """
        if not isinstance(p, cls):
            raise ValueError("p must be an FXTensor")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        result = p.copy()
        for _ in range(n - 1):
            result = result.tensor_product(p)
        return result

    @classmethod
    def mixture_of_iid(cls, prior, kernel, n):
        """Return ``prior ; copy_Θ^{(n)} ; kernel^{⊗n}``.

        ``prior`` is a state ``I → Θ`` and ``kernel`` is a kernel
        ``Θ → X``. The result is a state ``I → X^{⊗n}``: one draw from the
        prior, copied, then ``n`` independent samples from the kernel. It is
        exchangeable and, as ``n`` varies, Kolmogorov consistent. Codomain
        labels are the codomain labels of ``kernel``, repeated ``n`` times.

        Raises
        ------
        ValueError
            If ``prior`` is not a state, the profiles do not meet, or ``n``
            is not a positive integer.
        """
        if not isinstance(prior, cls) or not isinstance(kernel, cls):
            raise ValueError("prior and kernel must be FXTensor values")
        if prior._profile[0]:
            raise ValueError("prior must be a state I → Θ")
        if list(prior._profile[1]) != list(kernel._profile[0]):
            raise ValueError("prior codomain must match the domain of kernel")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if prior._labels is not None and prior._labels[1]:
            theta = prior._labels[1]
        else:
            theta = list(prior._profile[1])
        copied = prior.composition(cls.copy_tensor(theta, n))
        kernels = kernel.copy()
        for _ in range(n - 1):
            kernels = kernels.tensor_product(kernel)
        result = copied.composition(kernels)
        if (
            result._labels is None
            and kernel._labels is not None
            and kernel._labels[1]
        ):
            result = result._spawn(
                result._profile[0],
                result._profile[1],
                ([], list(kernel._labels[1]) * n),
                result.data,
            )
        return result

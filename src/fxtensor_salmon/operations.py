from __future__ import annotations
import numpy as np

from .profile import _constructor_profile


class OperationsMixin:
    def _spawn(self, domain, codomain, labels, data):
        return type(self)(_constructor_profile(domain, codomain, labels), data=data)

    def composition(self, other: 'FXTensor') -> 'FXTensor':
        """Standard tensor composition."""
        if self._profile[1] != other._profile[0]:
            raise ValueError("Tensors are not composable: codomain of self must match domain of other.")
        self_codomain_axes = list(range(len(self._profile[0]), self.data.ndim))
        other_domain_axes = list(range(len(other._profile[0])))
        new_domain = self._profile[0]
        new_codomain = other._profile[1]
        new_labels = None
        if self._labels is not None and other._labels is not None:
            new_labels = (self._labels[0], other._labels[1])
        result_data = np.tensordot(self.data, other.data, axes=(self_codomain_axes, other_domain_axes))
        return self._spawn(new_domain, new_codomain, new_labels, result_data)

    def tensor_product(self, other: 'FXTensor') -> 'FXTensor':
        """Perform tensor product operation."""
        new_domain = self._profile[0] + other._profile[0]
        new_codomain = self._profile[1] + other._profile[1]
        new_domain_labels = (self._labels[0] if self._labels is not None else []) + (other._labels[0] if other._labels is not None else [])
        new_codomain_labels = (self._labels[1] if self._labels is not None else []) + (other._labels[1] if other._labels is not None else [])
        new_labels = (new_domain_labels, new_codomain_labels) if new_domain_labels or new_codomain_labels else None
        # Mixing labeled and unlabeled factors cannot be represented; drop labels
        # (as composition does) unless the unlabeled side has no axes at all.
        if new_labels is not None and (
            (self._labels is None and self.data.ndim > 0)
            or (other._labels is None and other.data.ndim > 0)
        ):
            new_labels = None
        self_dom_len = len(self._profile[0])
        other_dom_len = len(other._profile[0])
        self_cod_len = len(self._profile[1])
        other_cod_len = len(other._profile[1])
        self_reshaped = self.data.reshape(self._profile[0] + [1]*other_dom_len + self._profile[1] + [1]*other_cod_len)
        other_reshaped = other.data.reshape([1]*self_dom_len + other._profile[0] + [1]*self_cod_len + other._profile[1])
        new_data = self_reshaped * other_reshaped
        return self._spawn(new_domain, new_codomain, new_labels, new_data)

    def is_markov(self) -> bool:
        """Check if the tensor is a Markov tensor.

        For every domain index the codomain slice must sum to 1 (or 0, so
        that zero rows produced by :meth:`conditionalization` are accepted).
        States (empty domain) always return ``False``; check
        ``np.isclose(state.data.sum(), 1)`` for a normalized state.
        """
        if not self._profile[0]:
            return False
        codomain_axes = tuple(range(len(self._profile[0]), self.data.ndim))
        sums = np.sum(self.data, axis=codomain_axes)
        rtol = type(self)._RTOL
        atol = type(self)._ATOL
        return np.all(np.isclose(sums, 1, rtol=rtol, atol=atol) | np.isclose(sums, 0, rtol=rtol, atol=atol))

    def conditionalization(self, concat_start_index: int) -> 'FXTensor':
        """Create a conditional probability distribution from a joint state."""
        if self._profile[0]:
            raise ValueError("Tensor must be a state (empty domain) for conditionalization")
        split_point = concat_start_index - 1
        if not (0 < concat_start_index <= len(self._profile[1])):
            raise ValueError("concat_start_index is out of bounds")
        new_domain = self._profile[1][:split_point]
        new_codomain = self._profile[1][split_point:]
        new_labels = None
        if self._labels is not None:
            new_domain_labels = self._labels[1][:split_point]
            new_codomain_labels = self._labels[1][split_point:]
            new_labels = (new_domain_labels, new_codomain_labels)
        sum_over_codomain = np.sum(self.data, axis=tuple(range(split_point, len(self._profile[1]))), keepdims=True)
        sum_over_codomain[sum_over_codomain == 0] = 1
        new_data = self.data / sum_over_codomain
        return self._spawn(new_domain, new_codomain, new_labels, new_data)

    def conditional(self, concat_start_index: int) -> 'FXTensor':
        """Conditional of a kernel ``f: A → X ⊗ Y`` as ``A ⊗ X → Y``.

        Generalizes :meth:`conditionalization` to tensors with a domain.
        ``concat_start_index`` (1-based) is the first codomain axis of
        ``Y``. Entries with ``f_X(x|a) = 0`` stay zero, as in
        :meth:`conditionalization`. For a state the result equals
        ``conditionalization(concat_start_index)``.
        """
        codomain_len = len(self._profile[1])
        if not (0 < concat_start_index <= codomain_len):
            raise ValueError("concat_start_index is out of bounds")
        split_point = concat_start_index - 1
        domain_len = len(self._profile[0])
        new_domain = self._profile[0] + self._profile[1][:split_point]
        new_codomain = self._profile[1][split_point:]
        new_labels = None
        if self._labels is not None:
            new_labels = (
                self._labels[0] + self._labels[1][:split_point],
                self._labels[1][split_point:],
            )
        sum_axes = tuple(range(domain_len + split_point, self.data.ndim))
        denom = np.sum(self.data, axis=sum_axes, keepdims=True)
        denom[denom == 0] = 1
        return self._spawn(new_domain, new_codomain, new_labels, self.data / denom)

    def bayesian_inversion(self, prior: 'FXTensor') -> 'FXTensor':
        """Bayesian inversion of ``f: X → Y`` with respect to ``prior: I → X``.

        Returns ``f†: Y → X`` with ``f†(x|y) = prior(x) f(y|x) / Σ_x' prior(x') f(y|x')``
        (Cho & Jacobs 2017). Rows with zero evidence stay zero. Labels are
        swapped from ``self`` (or taken from ``prior`` for ``X``).
        """
        if prior._profile[0]:
            raise ValueError("prior must be a state (empty domain)")
        if prior._profile[1] != self._profile[0]:
            raise ValueError("prior codomain must match the domain of self")
        x_len = len(self._profile[0])
        y_len = len(self._profile[1])
        prior_data = prior.data.reshape(self._profile[0] + [1] * y_len)
        joint = prior_data * self.data
        evidence = np.sum(joint, axis=tuple(range(x_len)), keepdims=True)
        evidence[evidence == 0] = 1
        order = list(range(x_len, x_len + y_len)) + list(range(x_len))
        new_data = (joint / evidence).transpose(order)
        x_labels = None
        if self._labels is not None and self._labels[0]:
            x_labels = self._labels[0]
        elif prior._labels is not None and prior._labels[1]:
            x_labels = prior._labels[1]
        y_labels = self._labels[1] if self._labels is not None and self._labels[1] else None
        new_labels = None
        if (x_labels is not None or not x_len) and (y_labels is not None or not y_len):
            new_labels = (y_labels or [], x_labels or [])
        return self._spawn(self._profile[1], self._profile[0], new_labels, new_data)

    def is_deterministic(self) -> bool:
        """Check whether every codomain slice is a point mass (one-hot).

        In finite stochastic matrices this is equivalent to Fritz's
        definition ``f ; copy = copy ; (f ⊗ f)`` for Markov kernels. Zero
        slices are not considered deterministic.
        """
        dom_size = int(np.prod(self._profile[0])) if self._profile[0] else 1
        rows = np.asarray(self.data, dtype=float).reshape(dom_size, -1)
        rtol = type(self)._RTOL
        atol = type(self)._ATOL
        zero = np.isclose(rows, 0, rtol=rtol, atol=atol)
        one = np.isclose(rows, 1, rtol=rtol, atol=atol)
        return bool(np.all(zero | one) and np.all(one.sum(axis=1) == 1))

    def support(self) -> 'FXTensor':
        """Indicator tensor of the nonzero entries (same profile and labels).

        For a state ``p: I → X`` this is the support ``{x : p(x) > 0}``; for a
        kernel ``f: A → X`` it is the support relation ``{(a, x) : f(x|a) > 0}``.
        Entries within ``_ATOL`` of zero count as zero.
        """
        rtol = type(self)._RTOL
        atol = type(self)._ATOL
        mask = ~np.isclose(self.data, 0, rtol=rtol, atol=atol)
        return self._spawn(self._profile[0], self._profile[1], self._labels, mask.astype(float))

    def almost_surely_equal(self, other: 'FXTensor', prior: 'FXTensor') -> bool:
        """Check ``self = other`` ``prior``-almost surely.

        For ``f, g: X → Y`` and a state ``p: I → X`` this is Fritz's
        ``p ; copy ; (id ⊗ f) = p ; copy ; (id ⊗ g)``; for finite tensors it
        means ``f(·|x) = g(·|x)`` for every ``x`` with ``p(x) > 0``. Only
        profiles are compared, not labels.
        """
        if self._profile != other._profile:
            raise ValueError("Tensors must have the same profile")
        if prior._profile[0]:
            raise ValueError("prior must be a state (empty domain)")
        if prior._profile[1] != self._profile[0]:
            raise ValueError("prior codomain must match the domain of self")
        weights = prior.data.reshape(self._profile[0] + [1] * len(self._profile[1]))
        return bool(np.allclose(
            weights * self.data,
            weights * other.data,
            rtol=type(self)._RTOL,
            atol=type(self)._ATOL,
        ))

    def is_absolutely_continuous(self, other: 'FXTensor') -> bool:
        """Check ``self ≪ other``: wherever ``other`` is zero, ``self`` is zero.

        For states this is the usual absolute continuity of distributions;
        for kernels ``f, g: A → X`` it holds input-wise, i.e. ``f(·|a) ≪ g(·|a)``
        for every ``a`` (finite case of Fritz et al., arXiv:2308.00651).
        Only profiles are compared, not labels.
        """
        if self._profile != other._profile:
            raise ValueError("Tensors must have the same profile")
        rtol = type(self)._RTOL
        atol = type(self)._ATOL
        other_zero = np.isclose(other.data, 0, rtol=rtol, atol=atol)
        self_zero = np.isclose(self.data, 0, rtol=rtol, atol=atol)
        return bool(np.all(self_zero | ~other_zero))

    def marginalization(self, start_B: int) -> 'FXTensor':
        """Marginalize out a part of the codomain by summing over it."""
        domain_len = len(self._profile[0])
        codomain_len = len(self._profile[1])
        if not (1 <= start_B <= codomain_len + 1):
            raise ValueError("start_B must be a valid split index in the codomain")
        if start_B > codomain_len:
            return self
        sum_axes = tuple(range(domain_len + start_B - 1, self.data.ndim))
        new_data = np.sum(self.data, axis=sum_axes)
        new_codomain = self._profile[1][:start_B - 1]
        new_labels = None
        if self._labels is not None:
            new_labels = (self._labels[0], self._labels[1][:start_B - 1])
        return self._spawn(self._profile[0], new_codomain, new_labels, new_data)

    def jointification(self, other: 'FXTensor') -> 'FXTensor':
        """Create a joint state from two tensors."""
        if self._profile[0] or other._profile[0]:
            raise ValueError("Both tensors must be states (empty domain) for jointification")
        self_expanded = self.data.reshape(self._profile[1] + [1] * len(other._profile[1]))
        result_data = self_expanded * other.data
        new_codomain = self._profile[1] + other._profile[1]
        new_labels = None
        if self._labels is not None and other._labels is not None:
            new_labels = ([], self._labels[1] + other._labels[1])
        return self._spawn([], new_codomain, new_labels, result_data)

    def partial_composition(self, other: 'FXTensor', concat_start_index: int) -> 'FXTensor':
        """Perform partial composition."""
        idx = concat_start_index - 1
        a_part = self._profile[1][:idx]
        b_part = self._profile[1][idx:]
        if other._profile[0][:len(b_part)] != b_part:
            raise ValueError("Tensors are not suitable for partial composition")
        id_a = type(self).identity_tensor(a_part)
        id_a_otimes_other = id_a.tensor_product(other)
        len_a = len(a_part)
        len_o_dom = len(other._profile[0])
        len_a_cod = len(a_part)
        transpose_order = list(range(len_a, len_a + len_o_dom)) + \
                          list(range(len_a)) + \
                          list(range(len_a + len_o_dom, len_a + len_o_dom + len_a_cod)) + \
                          list(range(len_a + len_o_dom + len_a_cod, id_a_otimes_other.data.ndim))
        transposed_data = id_a_otimes_other.data.transpose(transpose_order)
        new_codomain = a_part + other._profile[1]
        reshaped_other = type(self)([self._profile[1], new_codomain], data=transposed_data.reshape(tuple(self._profile[1] + new_codomain)))
        return self.composition(reshaped_other)

    def first_marginalization(self, concat_start_index: int) -> 'FXTensor':
        """Marginalize out the second part of the codomain."""
        start_B = concat_start_index - 1
        a_sizes = self._profile[1][:start_B]
        b_sizes = self._profile[1][start_B:]
        unit_a = type(self).identity_tensor(a_sizes)
        excl_b = type(self).exclamation(b_sizes)
        tp = unit_a.tensor_product(excl_b)
        return self.composition(tp)

    def second_marginalization(self, concat_start_index: int) -> 'FXTensor':
        """Marginalize out the first part of the codomain."""
        start_B = concat_start_index - 1
        a_sizes = self._profile[1][:start_B]
        b_sizes = self._profile[1][start_B:]
        excl_a = type(self).exclamation(a_sizes)
        unit_b = type(self).identity_tensor(b_sizes)
        tp = excl_a.tensor_product(unit_b)
        return self.composition(tp)

    def discard_prefix(self, start_B: int) -> 'FXTensor':
        """Discard a prefix of the codomain by summing over it."""
        domain_len = len(self._profile[0])
        codomain_len = len(self._profile[1])
        if not (1 <= start_B <= codomain_len + 1):
            raise ValueError("start_B must be a valid split index in the codomain")
        if start_B == 1:
            return self
        num_axes_to_sum = start_B - 1
        sum_axes = tuple(range(domain_len, domain_len + num_axes_to_sum))
        new_data = np.sum(self.data, axis=sum_axes)
        new_codomain = self._profile[1][num_axes_to_sum:]
        new_labels = None
        if self._labels is not None:
            new_labels = (self._labels[0], self._labels[1][num_axes_to_sum:])
        return self._spawn(self._profile[0], new_codomain, new_labels, new_data)

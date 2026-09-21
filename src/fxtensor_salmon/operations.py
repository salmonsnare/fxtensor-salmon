from __future__ import annotations
import numpy as np


class OperationsMixin:
    def composition(self, other: 'FXTensor') -> 'FXTensor':
        """Standard tensor composition."""
        if self._profile[1] != other._profile[0]:
            raise ValueError("Tensors are not composable: codomain of self must match domain of other.")
        self_codomain_axes = list(range(len(self._profile[0]), self.data.ndim))
        other_domain_axes = list(range(len(other._profile[0])))
        new_domain = self._profile[0]
        new_codomain = other._profile[1]
        new_profile = [new_domain, new_codomain]
        new_labels = None
        if self._labels is not None and other._labels is not None:
            new_labels = (self._labels[0], other._labels[1])
        result_data = np.tensordot(self.data, other.data, axes=(self_codomain_axes, other_domain_axes))
        if new_labels and (new_labels[0] or new_labels[1]):
            result_profile = [new_labels[0], new_labels[1]]
        else:
            result_profile = new_profile
        return type(self)(result_profile, data=result_data)

    def tensor_product(self, other: 'FXTensor') -> 'FXTensor':
        """Perform tensor product operation."""
        new_domain = self._profile[0] + other._profile[0]
        new_codomain = self._profile[1] + other._profile[1]
        new_profile = [new_domain, new_codomain]
        new_domain_labels = (self._labels[0] if self._labels is not None else []) + (other._labels[0] if other._labels is not None else [])
        new_codomain_labels = (self._labels[1] if self._labels is not None else []) + (other._labels[1] if other._labels is not None else [])
        new_labels = (new_domain_labels, new_codomain_labels) if new_domain_labels or new_codomain_labels else None
        self_dom_len = len(self._profile[0])
        other_dom_len = len(other._profile[0])
        self_cod_len = len(self._profile[1])
        other_cod_len = len(self._profile[1])
        self_reshaped = self.data.reshape(self._profile[0] + [1]*other_dom_len + self._profile[1] + [1]*other_cod_len)
        other_reshaped = other.data.reshape([1]*self_dom_len + other._profile[0] + [1]*self_cod_len + other._profile[1])
        new_data = self_reshaped * other_reshaped
        if new_labels and (new_labels[0] or new_labels[1]):
            result_profile = [new_labels[0], new_labels[1]]
        else:
            result_profile = new_profile
        return type(self)(result_profile, data=new_data)

    def is_markov(self) -> bool:
        """Check if the tensor is a Markov tensor."""
        if not self._profile[0]:
            return False
        codomain_axes = tuple(range(len(self._profile[0]), self.data.ndim))
        sums = np.sum(self.data, axis=codomain_axes)
        return np.all(np.isclose(sums, 1) | np.isclose(sums, 0))

    def conditionalization(self, concat_start_index: int) -> 'FXTensor':
        """Create a conditional probability distribution from a joint state."""
        if self._profile[0]:
            raise ValueError("Tensor must be a state (empty domain) for conditionalization")
        split_point = concat_start_index - 1
        if not (0 < concat_start_index <= len(self._profile[1])):
            raise ValueError("concat_start_index is out of bounds")
        new_domain = self._profile[1][:split_point]
        new_codomain = self._profile[1][split_point:]
        new_profile = [new_domain, new_codomain]
        new_labels = None
        if self._labels is not None:
            new_domain_labels = self._labels[1][:split_point]
            new_codomain_labels = self._labels[1][split_point:]
            new_labels = (new_domain_labels, new_codomain_labels)
        sum_over_codomain = np.sum(self.data, axis=tuple(range(split_point, len(self._profile[1]))), keepdims=True)
        sum_over_codomain[sum_over_codomain == 0] = 1
        new_data = self.data / sum_over_codomain
        if new_labels and (new_labels[0] or new_labels[1]):
            result_profile = [new_labels[0], new_labels[1]]
        else:
            result_profile = new_profile
        return type(self)(result_profile, data=new_data)

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
        new_profile = [self._profile[0], new_codomain]
        new_labels = None
        if self._labels is not None:
            new_labels = (self._labels[0], self._labels[1][:start_B - 1])
        if new_labels and (new_labels[0] or new_labels[1]):
            result_profile = [new_labels[0], new_labels[1]]
        else:
            result_profile = new_profile
        return type(self)(result_profile, data=new_data)

    def jointification(self, other: 'FXTensor') -> 'FXTensor':
        """Create a joint state from two tensors."""
        if self._profile[0] or other._profile[0]:
            raise ValueError("Both tensors must be states (empty domain) for jointification")
        self_expanded = self.data.reshape(self._profile[1] + [1] * len(other._profile[1]))
        result_data = self_expanded * other.data
        new_codomain = self._profile[1] + other._profile[1]
        new_profile = [[], new_codomain]
        new_labels = None
        if self._labels is not None and other._labels is not None:
            new_labels = ([], self._labels[1] + other._labels[1])
        if new_labels and (new_labels[0] or new_labels[1]):
            result_profile = [new_labels[0], new_labels[1]]
        else:
            result_profile = new_profile
        return type(self)(result_profile, data=result_data)

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
        new_profile = [self._profile[0], new_codomain]
        new_labels = None
        if self._labels is not None:
            new_labels = (self._labels[0], self._labels[1][num_axes_to_sum:])
        if new_labels and (new_labels[0] or new_labels[1]):
            result_profile = [new_labels[0], new_labels[1]]
        else:
            result_profile = new_profile
        return type(self)(result_profile, data=new_data)

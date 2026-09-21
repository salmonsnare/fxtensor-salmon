from __future__ import annotations
import numpy as np
from typing import List, Union


class SpecialMixin:
    @classmethod
    def swap(cls, list_a: List[int], list_b: List[int]) -> 'FXTensor':
        """Create a swap tensor."""
        domain = list_a + list_b
        codomain = list_b + list_a
        profile = [domain, codomain]
        shape = tuple(domain + codomain)
        data = np.zeros(shape)
        len_a = len(list_a)
        for idx in np.ndindex(*domain):
            idx_a = idx[:len_a]
            idx_b = idx[len_a:]
            to_idx = idx_b + idx_a
            full_idx = tuple(list(idx) + list(to_idx))
            data[full_idx] = 1
        return cls(profile, data=data)

    @classmethod
    def exclamation(cls, list_x: List[int]) -> 'FXTensor':
        """Create an exclamation tensor (discarding)."""
        profile = [list_x, []]
        data = np.ones(tuple(list_x))
        return cls(profile, data=data)

    @classmethod
    def identity_tensor(cls, list_x: Union[List[int], List[List[str]]]) -> 'FXTensor':
        """Create an identity tensor."""
        if not list_x:
            return cls([[], []], data=np.array(1.0))
        if isinstance(list_x[0], int):
            dims = list_x
        else:
            dims = [len(dim) for dim in list_x]
        data = np.eye(np.prod(dims)).reshape(dims + dims)
        if isinstance(list_x[0], list):
            return cls([list_x, list_x], data=data)
        else:
            return cls([dims, dims], data=data)

    @classmethod
    def unit_tensor(cls, dims: List[int]) -> 'FXTensor':
        """Create a unit tensor (all ones) with the given dimensions."""
        return cls([[], dims], data=np.ones(tuple(dims)))

    @classmethod
    def delta_tensor(cls, dims: List[int]) -> 'FXTensor':
        """Create a delta tensor (identity matrix) with the given dimensions."""
        return cls.identity_tensor(dims)

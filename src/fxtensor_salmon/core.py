from __future__ import annotations
import numpy as np
from typing import List, Tuple, Union, Optional

from .io import IOMixin
from .operations import OperationsMixin
from .special import SpecialMixin


class FXTensor(IOMixin, OperationsMixin, SpecialMixin):
    """A tensor with support for string-labeled dimensions and tensor operations."""

    def __init__(
        self,
        profile: Union[
            List[List[Union[int, List[str]]]],
            Tuple[List[int], List[int]]
        ],
        data: Optional[np.ndarray] = None,
    ) -> None:
        if isinstance(profile, tuple):
            domain_dims = list(profile[0])
            codomain_dims = list(profile[1])
            self._labels = None
        else:
            if len(profile) != 2:
                raise ValueError("Invalid profile format")
            is_numeric = True
            for group in profile:
                if group:
                    if not isinstance(group[0], int):
                        is_numeric = False
                    break
            if is_numeric:
                domain_dims = list(profile[0])
                codomain_dims = list(profile[1])
                self._labels = None
            else:
                domain_labels = profile[0] if profile[0] else []
                codomain_labels = profile[1] if profile[1] else []
                self._labels = (domain_labels, codomain_labels)
                domain_dims = [len(dim) for dim in domain_labels]
                codomain_dims = [len(dim) for dim in codomain_labels]
        self._profile = (domain_dims, codomain_dims)

        shape = tuple(domain_dims + codomain_dims)
        if data is not None:
            self.data = np.asarray(data)
            if self.data.shape != shape:
                raise ValueError(f"Data shape {data.shape} does not match profile {shape}")
        else:
            self.data = np.zeros(shape)

    @property
    def profile(self) -> List[List[int]]:
        """Get the tensor's profile as a list of lists [[domain], [codomain]]."""
        return [list(self._profile[0]), list(self._profile[1])]

    @property
    def labels(self) -> Optional[Tuple[List[List[str]], List[List[str]]]]:
        """Get the tensor's string labels."""
        domain = self._labels[0] if self._labels and self._labels[0] else None
        codomain = self._labels[1] if self._labels and self._labels[1] else None
        return (domain, codomain)

    def get_label_index(self, axis: int, label: str) -> int:
        """
        Get the index of a label for a given axis.
        """
        if self._labels is None:
            raise ValueError("No labels defined for this tensor")
        domain_len = len(self._profile[0])
        if axis < domain_len:
            return self._labels[0][axis].index(label)
        else:
            return self._labels[1][axis - domain_len].index(label)

    def get_index_label(self, axis: int, index: int) -> str:
        """
        Get the label for a given axis and index.
        """
        if self._labels is None:
            raise ValueError("No labels defined for this tensor")
        domain_len = len(self._profile[0])
        if axis < domain_len:
            return self._labels[0][axis][index]
        else:
            return self._labels[1][axis - domain_len][index]

    def __repr__(self) -> str:
        """Return a string representation of the FXTensor."""
        if self._labels is not None:
            domain_labels = f"[{', '.join(['[' + ', '.join(dim) + ']' for dim in self._labels[0]])}]" if self._labels[0] else "[]"
            codomain_labels = f"[{', '.join(['[' + ', '.join(dim) + ']' for dim in self._labels[1]])}]" if self._labels[1] else "[]"
            return f"FXTensor(profile={self.profile}, labels=({domain_labels}, {codomain_labels}), shape={self.data.shape})"
        else:
            return f"FXTensor(profile={self.profile}, shape={self.data.shape})"

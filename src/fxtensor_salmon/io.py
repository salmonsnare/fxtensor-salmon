from __future__ import annotations
import numpy as np
import json
import ast
from typing import List, Tuple, Dict, Any, Union
from fractions import Fraction


class IOMixin:
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'FXTensor':
        """Create a tensor from JSON data."""
        profile = json_data["profile"]
        if len(profile) != 2:
            raise ValueError("Invalid profile format")
        is_numeric = True
        for group in profile:
            if group:
                if not isinstance(group[0], int):
                    is_numeric = False
                break
        if is_numeric:
            shape = tuple(profile[0] + profile[1]) if profile else ()
        else:
            domain_dims = [len(dim) for dim in profile[0]] if profile[0] else []
            codomain_dims = [len(dim) for dim in profile[1]] if profile[1] else []
            shape = tuple(domain_dims + codomain_dims)
        data = np.array(json_data["data"]).reshape(shape)
        return cls(profile, data=data)

    @classmethod
    def from_strands(
        cls, 
        profile: Union[Tuple[List[int], List[int]], List[List[List[str]]]],
        strands: Dict[str, Union[float, Fraction]]
    ) -> 'FXTensor':
        """Create a tensor from strands (string representation)."""
        tensor = cls(profile)
        is_labeled = tensor._labels is not None
        for strand_str, weight in strands.items():
            domain_part, codomain_part = ast.literal_eval(strand_str)
            indices = []
            for part, label_idx in [(domain_part, 0), (codomain_part, 1)]:
                for axis, dim_part in enumerate(part):
                    val = dim_part[0]
                    if is_labeled:
                        idx = tensor._labels[label_idx][axis].index(val)
                    else:
                        idx = val - 1
                    indices.append(idx)
            tensor.data[tuple(indices)] = float(weight)
        return tensor

    def to_json(self) -> dict:
        """Convert the tensor to a JSON-serializable dictionary."""
        if self._labels is not None:
            profile = [self._labels[0], self._labels[1]]
        else:
            profile = [list(self._profile[0]), list(self._profile[1])]
        data = self.data.tolist()
        return {
            "profile": profile,
            "data": data,
            "dtype": str(self.data.dtype)
        }

    def save_to_file(self, filename: str) -> None:
        """Save the tensor to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> 'FXTensor':
        """Load a tensor from a JSON file."""
        with open(filename, 'r') as f:
            return cls.from_json(json.load(f))

    def to_original_dict(self) -> Dict[str, Any]:
        """Convert tensor to original dictionary format with 1-based indexing."""
        strands = {}
        domain_len = len(self._profile[0])
        for idx in np.ndindex(self.data.shape):
            weight = self.data[idx]
            if weight != 0:
                domain_idx = [d + 1 for d in idx[:domain_len]]
                codomain_idx = [c + 1 for c in idx[domain_len:]]
                strand_key = str([domain_idx, codomain_idx])
                strands[strand_key] = weight
        return {"profile": self.profile, "strands": strands}

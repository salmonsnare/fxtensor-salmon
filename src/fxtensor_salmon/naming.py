"""Axis names for FXTensor: permute, marginal, and observe.

Names are variable names on wires. They are not the labels of the points
on a wire. A tensor constructed with ``names=None`` stays unnamed, and the
numeric operations are unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


NamePair = Tuple[Tuple[str, ...], Tuple[str, ...]]


def _as_name_tuple(raw, length: int, side: str) -> Tuple[str, ...]:
    if raw is None:
        if length == 0:
            return ()
        raise ValueError(f"{side} names are missing")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ValueError(f"{side} names must be a sequence of strings")
    if len(raw) != length:
        raise ValueError(f"{side} names must have length {length}, got {len(raw)}")
    names = []
    for name in raw:
        if not isinstance(name, str) or name == "":
            raise ValueError("each axis name must be a non-empty string")
        names.append(name)
    return tuple(names)


def _normalize_names(domain_len: int, codomain_len: int, names) -> Optional[NamePair]:
    """Return stored names, or ``None`` when the tensor is unnamed.

    ``names`` is ``None`` or ``(domain names, codomain names)``. Every name
    in the tensor must be unique.
    """
    if names is None:
        return None
    if isinstance(names, (str, bytes)) or not isinstance(names, (list, tuple)) or len(names) != 2:
        raise ValueError("names must be None or (domain names, codomain names)")
    domain = _as_name_tuple(names[0], domain_len, "domain")
    codomain = _as_name_tuple(names[1], codomain_len, "codomain")
    merged = domain + codomain
    if len(merged) != len(set(merged)):
        raise ValueError("names must be unique")
    if not merged:
        return None
    return (domain, codomain)


def _pack(domain, codomain) -> Optional[NamePair]:
    domain = tuple(domain)
    codomain = tuple(codomain)
    if not domain and not codomain:
        return None
    return (domain, codomain)


def _composition_names(left, right, left_dom_len: int, right_cod_len: int) -> Optional[NamePair]:
    """Names of ``left ; right``.

    When both the codomain names of ``left`` and the domain names of
    ``right`` are present, they must agree. The result takes the domain
    names of ``left`` and the codomain names of ``right``. A missing side
    contributes names only when that side has no axes.
    """
    left_cod = left[1] if left is not None else None
    right_dom = right[0] if right is not None else None
    if left_cod is not None and right_dom is not None and tuple(left_cod) != tuple(right_dom):
        raise ValueError("codomain names of self must match domain names of other")
    if left is None:
        if left_dom_len:
            return None
        out_dom: Tuple[str, ...] = ()
    else:
        out_dom = tuple(left[0])
    if right is None:
        if right_cod_len:
            return None
        out_cod: Tuple[str, ...] = ()
    else:
        out_cod = tuple(right[1])
    return _pack(out_dom, out_cod)


def _tensor_names(left, right, left_axes: int, right_axes: int) -> Optional[NamePair]:
    """Concatenate names. Duplicates raise ``ValueError``.

    If either factor has axes but no names, the product is unnamed: there is
    no name list to concatenate. A factor with no axes does not add names.
    """
    if (left is None and left_axes) or (right is None and right_axes):
        return None
    domain = (tuple(left[0]) if left is not None else ()) + (tuple(right[0]) if right is not None else ())
    codomain = (tuple(left[1]) if left is not None else ()) + (tuple(right[1]) if right is not None else ())
    merged = domain + codomain
    if len(merged) != len(set(merged)):
        raise ValueError("tensor_product names must be unique")
    return _pack(domain, codomain)


def _swapped_names(names) -> Optional[NamePair]:
    """Swap domain and codomain names."""
    if names is None:
        return None
    return _pack(names[1], names[0])


def _split_codomain_names(names, split_point: int) -> Optional[NamePair]:
    """Move a codomain prefix onto the domain, as in ``conditional``."""
    if names is None:
        return None
    return _pack(tuple(names[0]) + tuple(names[1][:split_point]), names[1][split_point:])


def _keep_codomain_prefix(names, keep: int) -> Optional[NamePair]:
    """Keep domain names and the first ``keep`` codomain names."""
    if names is None:
        return None
    return _pack(names[0], names[1][:keep])


def _keep_codomain_suffix(names, drop: int) -> Optional[NamePair]:
    """Keep domain names and the codomain names after ``drop`` axes."""
    if names is None:
        return None
    return _pack(names[0], names[1][drop:])


def _require_names(tensor) -> NamePair:
    if tensor._names is None:
        raise ValueError("tensor has no axis names")
    return tensor._names


def _evidence_index(tensor, codomain_axis: int, value) -> int:
    labels = None
    if tensor._labels is not None and tensor._labels[1]:
        labels = tensor._labels[1][codomain_axis]
    if labels:
        if not isinstance(value, str):
            raise ValueError("evidence must be a label string")
        try:
            return labels.index(value)
        except ValueError:
            raise ValueError(f"unknown label {value!r}") from None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("evidence must be an integer index when the axis has no labels")
    size = tensor._profile[1][codomain_axis]
    if not 0 <= value < size:
        raise ValueError("evidence index out of range")
    return value


class NamingMixin:
    """Name-based permute, marginal, and observe."""

    def rename(self, **mapping: str) -> "FXTensor":
        """Return a copy with axis names replaced.

        Each keyword is an existing name and its value is the new name.
        Names that are not mentioned stay. The result must still be unique.
        """
        current = _require_names(self)
        known = set(current[0] + current[1])
        for old, new in mapping.items():
            if old not in known:
                raise ValueError(f"unknown name {old!r}")
            if not isinstance(new, str) or new == "":
                raise ValueError("each axis name must be a non-empty string")

        def _map(name: str) -> str:
            return mapping.get(name, name)

        return self._spawn(
            self._profile[0],
            self._profile[1],
            self._labels,
            np.array(self.data, copy=True),
            names=(tuple(_map(name) for name in current[0]), tuple(_map(name) for name in current[1])),
        )

    def axis(self, name: str) -> int:
        """Return the axis index of ``name``.

        Domain axes are counted first, then codomain axes, as in
        :meth:`get_label_index`.
        """
        current = _require_names(self)
        try:
            return (current[0] + current[1]).index(name)
        except ValueError:
            raise ValueError(f"unknown name {name!r}") from None

    def permute(self, order: Sequence[str]) -> "FXTensor":
        """Reorder the codomain axes into ``order``.

        The domain is unchanged. Labels move with their axes. ``order`` must
        be a permutation of the codomain names. For a state the codomain is
        the whole profile, so this reorders every axis.
        """
        current = _require_names(self)
        codomain = list(current[1])
        requested = list(order)
        if len(requested) != len(set(requested)) or set(requested) != set(codomain):
            raise ValueError("order must be a permutation of the codomain names")
        indices = [codomain.index(name) for name in requested]
        domain_len = len(self._profile[0])
        new_data = np.transpose(
            self.data,
            list(range(domain_len)) + [domain_len + index for index in indices],
        )
        new_codomain = [self._profile[1][index] for index in indices]
        new_labels = None
        if self._labels is not None:
            new_labels = (self._labels[0], [self._labels[1][index] for index in indices])
        new_names = (current[0], tuple(requested))
        return self._spawn(self._profile[0], new_codomain, new_labels, new_data, new_names)

    def marginal(self, *keep: str) -> "FXTensor":
        """Keep the named codomain axes, in the order given, and sum out the rest.

        A middle axis can be kept or dropped. For a state ``p: I → A ⊗ B ⊗ C``,
        ``p.marginal("B")`` equals ``p ; (! ⊗ id_B ⊗ !)``: the deterministic
        wiring that discards every codomain wire except ``B``. Labels move
        with the kept axes. The domain is unchanged.
        """
        current = _require_names(self)
        codomain = list(current[1])
        if len(keep) != len(set(keep)):
            raise ValueError("duplicate name in keep")
        missing = [name for name in keep if name not in codomain]
        if missing:
            raise ValueError(f"unknown codomain name {missing[0]!r}")
        keep_idx = [codomain.index(name) for name in keep]
        return self._marginal_indices(keep_idx)

    def marginalize(self, drop: Sequence[str]) -> "FXTensor":
        """Sum out the named codomain axes in ``drop``.

        The remaining codomain axes stay in their original order. This is
        ``marginal`` of the complement of ``drop``.
        """
        if isinstance(drop, (str, bytes)) or not isinstance(drop, (list, tuple)):
            raise ValueError("drop must be a sequence of names")
        current = _require_names(self)
        codomain = list(current[1])
        if len(drop) != len(set(drop)):
            raise ValueError("duplicate name in drop")
        missing = [name for name in drop if name not in codomain]
        if missing:
            raise ValueError(f"unknown codomain name {missing[0]!r}")
        dropped = set(drop)
        keep = [name for name in codomain if name not in dropped]
        return self.marginal(*keep)

    def _marginal_indices(self, keep_idx: Sequence[int]) -> "FXTensor":
        current = self._names
        domain_len = len(self._profile[0])
        codomain_len = len(self._profile[1])
        keep_set = set(keep_idx)
        drop_idx = [index for index in range(codomain_len) if index not in keep_set]
        sum_axes = tuple(domain_len + index for index in drop_idx)
        if sum_axes:
            reduced = np.sum(self.data, axis=sum_axes)
        else:
            reduced = np.array(self.data, copy=True)
        remaining = [index for index in range(codomain_len) if index not in set(drop_idx)]
        order = [remaining.index(index) for index in keep_idx]
        perm = list(range(domain_len)) + [domain_len + slot for slot in order]
        if reduced.ndim == 0 or not perm:
            new_data = np.asarray(reduced, dtype=float)
        else:
            new_data = np.transpose(reduced, perm)
        new_codomain = [self._profile[1][index] for index in keep_idx]
        new_labels = None
        if self._labels is not None:
            new_labels = (self._labels[0], [self._labels[1][index] for index in keep_idx])
        new_names = (tuple(current[0]), tuple(current[1][index] for index in keep_idx))
        return self._spawn(self._profile[0], new_codomain, new_labels, new_data, new_names)

    def observe(self, **evidence) -> Tuple["FXTensor", float]:
        """Condition a state on named outcomes.

        ``evidence`` maps an axis name to a label, or to an integer index
        when that axis has no labels. The result is ``(posterior, probability)``:
        the posterior is the joint state on the remaining axes, in their
        original order, and the probability is the evidence. Observing
        several axes at once matches observing them one at a time; the
        probability is then the joint probability of all of the outcomes.

        The posterior is the row of :meth:`conditionalization` after
        :meth:`permute` moves the observed wires to the front. Evidence 0
        raises ``ValueError``.

        Raises
        ------
        ValueError
            If ``self`` is not a state, a name or label is unknown, or the
            evidence has probability 0.
        """
        if self._profile[0]:
            raise ValueError("observe requires a state (empty domain)")
        current = _require_names(self)
        if not evidence:
            raise ValueError("observe requires at least one outcome")
        codomain = list(current[1])
        observed = list(evidence)
        missing = [name for name in observed if name not in codomain]
        if missing:
            raise ValueError(f"unknown codomain name {missing[0]!r}")
        rest = [name for name in codomain if name not in set(observed)]
        permuted = self.permute(observed + rest)
        indexes = [
            _evidence_index(permuted, position, evidence[name])
            for position, name in enumerate(observed)
        ]
        sliced = np.asarray(permuted.data[tuple(indexes)], dtype=float)
        probability = float(sliced.sum())
        if np.isclose(probability, 0.0, rtol=type(self)._RTOL, atol=type(self)._ATOL):
            raise ValueError("evidence has probability 0")
        posterior_data = sliced / probability
        new_labels = None
        if permuted._labels is not None:
            new_labels = ([], list(permuted._labels[1][len(observed) :]))
        new_codomain = list(permuted._profile[1][len(observed) :])
        new_names = ((), tuple(rest))
        posterior = self._spawn([], new_codomain, new_labels, posterior_data, new_names)
        return posterior, probability

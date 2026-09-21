from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .core import FXTensor

try:
    from discopy.cat import Category, Ob
    from discopy.markov import Box, Copy, Diagram, Discard, Functor, Merge, Swap, Ty
    from discopy.tensor import Dim, Tensor
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "DisCoPy integration requires the optional extra. "
        "Install with: pip install fxtensor-salmon[discopy]"
    ) from exc


__all__ = [
    "to_tensor",
    "from_tensor",
    "to_box",
    "to_diagram",
    "from_box",
    "from_diagram",
    "FXTensorFunctor",
]


ObImage = Union[int, Sequence[str], Dim]
ArImage = Union[FXTensor, Tensor, np.ndarray, Sequence[Any]]
Labels = Optional[Tuple[List[List[str]], List[List[str]]]]


def to_tensor(tensor: FXTensor) -> Tensor:
    """Convert an ``FXTensor`` to a ``discopy.tensor.Tensor``.

    Axis order is domain factors then codomain factors, matching
    ``Tensor.array`` as ``dom.inside + cod.inside``. Empty domain or
    codomain becomes the monoidal unit ``Dim()``. Labels are dropped;
    restore them with :func:`from_tensor` or use :func:`to_box`.
    """
    domain_dims, codomain_dims = tensor._profile
    dom = Dim(*domain_dims) if domain_dims else Dim()
    cod = Dim(*codomain_dims) if codomain_dims else Dim()
    return Tensor(np.asarray(tensor.data), dom, cod)


def from_tensor(tensor: Tensor, labels: Labels = None) -> FXTensor:
    """Convert a ``discopy.tensor.Tensor`` to an ``FXTensor``.

    ``labels`` is an optional ``(domain_labels, codomain_labels)`` pair
    in the same format as ``FXTensor.labels``. Without it the result is
    unlabeled and uses ``tensor.dom.inside`` / ``tensor.cod.inside``.
    """
    domain_dims = list(tensor.dom.inside)
    codomain_dims = list(tensor.cod.inside)
    data = np.asarray(tensor.array)
    profile = _profile_from_dims(domain_dims, codomain_dims, labels)
    return FXTensor(profile, data=data)


def to_box(tensor: FXTensor, name: str = "FXTensor") -> Box:
    """Wrap an ``FXTensor`` as a ``discopy.markov.Box``.

    Each profile factor becomes one wire. Unlabeled factor ``n`` is named
    ``n``; labeled factor ``['Sunny', 'Rainy']`` is stored as a JSON list
    on the type name so :func:`from_box` can restore it.
    """
    domain_dims, codomain_dims = tensor._profile
    domain_labels, codomain_labels = _label_groups(tensor)
    dom = _factors_to_ty(domain_dims, domain_labels)
    cod = _factors_to_ty(codomain_dims, codomain_labels)
    return Box(name, dom, cod, data=np.array(tensor.data, copy=True))


def to_diagram(tensor: FXTensor, name: str = "FXTensor") -> Diagram:
    """Wrap an ``FXTensor`` as a Markov diagram (a single box)."""
    return to_box(tensor, name=name)


def from_box(box: Box) -> FXTensor:
    """Convert a Markov box with array data to an ``FXTensor``.

    Dimensions come from ``box.data`` (reshaped to ``dom`` then ``cod``)
    and labels are restored from type names produced by :func:`to_box`.
    """
    if box.data is None:
        raise ValueError("Box has no data")
    data = np.asarray(box.data)
    domain_dims, domain_labels = _ty_to_factors(box.dom, _shape_prefix(data, 0, len(box.dom)))
    codomain_dims, codomain_labels = _ty_to_factors(
        box.cod, _shape_prefix(data, len(box.dom), len(box.cod))
    )
    expected = tuple(domain_dims + codomain_dims)
    if expected and data.shape != expected:
        if data.size != int(np.prod(expected)):
            raise ValueError(
                f"Box data shape {data.shape} does not match profile {expected}"
            )
        data = data.reshape(expected)
    elif not expected:
        data = np.asarray(data).reshape(())
    labels = None
    if domain_labels is not None or codomain_labels is not None:
        labels = (domain_labels or [], codomain_labels or [])
    profile = _profile_from_dims(domain_dims, codomain_dims, labels)
    return FXTensor(profile, data=data)


def from_diagram(
    diagram: Diagram,
    ob: Optional[Mapping[Ty, ObImage]] = None,
    ar: Optional[Union[Mapping[Box, ArImage], Callable[[Box], ArImage]]] = None,
) -> FXTensor:
    """Evaluate a Markov diagram as an ``FXTensor``.

    A single box with data is converted directly. Composite diagrams are
    interpreted by :class:`FXTensorFunctor`. Object images are inferred
    from encoded type names when ``ob`` is omitted.
    """
    if _is_data_box(diagram):
        return from_box(diagram)
    if len(getattr(diagram, "inside", ())) == 1 and _is_data_box(diagram.inside[0]):
        return from_box(diagram.inside[0])
    return FXTensorFunctor(ob or _infer_ob(diagram), ar)(diagram)


class FXTensorFunctor:
    """Interpret ``discopy.markov`` diagrams as ``FXTensor`` morphisms.

    Copy, Discard, Swap and Id are sent to :meth:`FXTensor.copy_tensor`,
    :meth:`FXTensor.exclamation`, :meth:`FXTensor.swap` and
    :meth:`FXTensor.identity_tensor`. Custom boxes are read from ``ar``
    (an ``FXTensor``, a DisCoPy ``Tensor``, or an array).

    Parameters
    ----------
    ob
        Atomic ``Ty`` to a dimension (``int``), a label list, or a ``Dim``.
    ar
        Generating ``Box`` to an ``FXTensor`` or array. Copy / Discard /
        Swap / Id are interpreted automatically and need no entry.
    """

    def __init__(
        self,
        ob: Union[Mapping[Ty, ObImage], Callable[[Ob], ObImage]],
        ar: Optional[Union[Mapping[Box, ArImage], Callable[[Box], ArImage]]] = None,
    ) -> None:
        self.ob = ob
        self.ar = {} if ar is None else ar
        ob_dim = _map_ob_to_dim(ob)
        ar_tensor = _map_ar_to_tensor(self.ar)
        self._functor = Functor(ob_dim, ar_tensor, cod=Category(Dim, Tensor[float]))

    def __call__(self, other: Any) -> Union[FXTensor, Dim, Ty]:
        result = self._functor(other)
        if isinstance(result, Tensor):
            labels = None
            if hasattr(other, "dom") and hasattr(other, "cod"):
                labels = self._labels_for(other.dom, other.cod)
            return from_tensor(result, labels=labels)
        return result

    def _labels_for(self, dom: Ty, cod: Ty) -> Labels:
        domain_labels = self._labels_for_ty(dom)
        codomain_labels = self._labels_for_ty(cod)
        if domain_labels is None or codomain_labels is None:
            return None
        if not domain_labels and not codomain_labels:
            return None
        return (domain_labels, codomain_labels)

    def _labels_for_ty(self, ty: Ty) -> Optional[List[List[str]]]:
        groups: List[List[str]] = []
        for atom in ty.inside:
            image = self._ob_image(atom)
            if not _is_label_list(image):
                return None
            groups.append([str(x) for x in image])
        return groups

    def _ob_image(self, atom: Ob) -> ObImage:
        key = atom if isinstance(atom, Ty) else Ty(atom)
        if callable(self.ob) and not isinstance(self.ob, Mapping):
            try:
                return self.ob(key)
            except (KeyError, TypeError):
                return self.ob(atom)
        mapping = self.ob
        if key in mapping:
            return mapping[key]
        if atom in mapping:
            return mapping[atom]
        raise KeyError(key)


def _label_groups(tensor: FXTensor) -> Tuple[Optional[List[List[str]]], Optional[List[List[str]]]]:
    if tensor._labels is None:
        return None, None
    domain = tensor._labels[0] if tensor._labels[0] else None
    codomain = tensor._labels[1] if tensor._labels[1] else None
    return domain, codomain


def _profile_from_dims(
    domain_dims: List[int],
    codomain_dims: List[int],
    labels: Labels,
) -> Union[List[List[int]], List[List[List[str]]]]:
    if labels is None:
        return [list(domain_dims), list(codomain_dims)]
    domain_labels, codomain_labels = labels
    domain_labels = domain_labels or []
    codomain_labels = codomain_labels or []
    if domain_labels:
        actual = [len(group) for group in domain_labels]
        if actual != list(domain_dims):
            raise ValueError(
                f"Domain labels {domain_labels} do not match dims {domain_dims}"
            )
    if codomain_labels:
        actual = [len(group) for group in codomain_labels]
        if actual != list(codomain_dims):
            raise ValueError(
                f"Codomain labels {codomain_labels} do not match dims {codomain_dims}"
            )
    if domain_labels or codomain_labels:
        return [domain_labels, codomain_labels]
    return [list(domain_dims), list(codomain_dims)]


def _encode_factor(dim: int, labels: Optional[Sequence[str]]) -> str:
    if labels is not None:
        return json.dumps([str(x) for x in labels], ensure_ascii=False, separators=(",", ":"))
    return str(int(dim))


def _decode_factor(name: str, dim_from_shape: Optional[int]) -> Tuple[Optional[int], Optional[List[str]]]:
    if name.isdigit():
        return int(name), None
    try:
        value = json.loads(name)
    except json.JSONDecodeError:
        return dim_from_shape, None
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return len(value), list(value)
    return dim_from_shape, None


def _factors_to_ty(dims: Sequence[int], labels: Optional[Sequence[Sequence[str]]]) -> Ty:
    ty = Ty()
    for index, dim in enumerate(dims):
        group = None if labels is None else labels[index]
        ty = ty @ Ty(_encode_factor(dim, group))
    return ty


def _shape_prefix(data: np.ndarray, start: int, count: int) -> Optional[Tuple[int, ...]]:
    if count == 0:
        return ()
    if data.ndim < start + count:
        return None
    return tuple(int(size) for size in data.shape[start:start + count])


def _ty_to_factors(
    ty: Ty,
    shape_prefix: Optional[Sequence[int]],
) -> Tuple[List[int], Optional[List[List[str]]]]:
    dims: List[int] = []
    labels: List[Optional[List[str]]] = []
    for index, atom in enumerate(ty.inside):
        dim_from_shape = None
        if shape_prefix is not None and index < len(shape_prefix):
            dim_from_shape = int(shape_prefix[index])
        dim, group = _decode_factor(atom.name, dim_from_shape)
        if dim is None:
            raise ValueError(f"Cannot infer dimension of wire {atom}")
        dims.append(dim)
        labels.append(group)
    if not dims:
        return [], []
    if any(group is None for group in labels):
        return dims, None
    return dims, [group for group in labels if group is not None]


def _is_label_list(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, str) for item in value)
    )


def _as_dim(value: ObImage) -> Dim:
    if isinstance(value, Dim):
        return value
    if isinstance(value, int):
        return Dim(value)
    if _is_label_list(value):
        return Dim(len(value))
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, int) for item in value):
        return Dim(*value)
    raise TypeError(f"Cannot interpret {value!r} as a DisCoPy Dim")


def _map_ob_to_dim(
    ob: Union[Mapping[Ty, ObImage], Callable[[Ob], ObImage]],
) -> Union[Dict[Ty, Dim], Callable[[Ob], Dim]]:
    if callable(ob) and not isinstance(ob, Mapping):
        return lambda atom: _as_dim(ob(atom))
    return {key: _as_dim(value) for key, value in ob.items()}


def _arrow_to_tensor(value: ArImage) -> np.ndarray:
    if isinstance(value, Tensor):
        return np.asarray(value.array, dtype=float)
    if isinstance(value, FXTensor):
        return np.asarray(value.data, dtype=float)
    return np.asarray(value, dtype=float)


def _map_ar_to_tensor(
    ar: Union[Mapping[Box, ArImage], Callable[[Box], ArImage]],
) -> Callable[[Box], Any]:
    if callable(ar) and not isinstance(ar, Mapping):
        return lambda box: _arrow_to_tensor(ar(box))
    converted = {box: _arrow_to_tensor(value) for box, value in ar.items()}

    def lookup(box: Box) -> Any:
        if box in converted:
            return converted[box]
        if box.data is not None:
            return np.asarray(box.data)
        raise KeyError(box)

    return lookup


def _is_data_box(other: Any) -> bool:
    return isinstance(other, Box) and not isinstance(other, (Copy, Merge, Swap, Discard)) and other.data is not None


def _infer_ob(diagram: Diagram) -> Dict[Ty, ObImage]:
    ob: Dict[Ty, ObImage] = {}
    types = [diagram.dom, diagram.cod]
    for box in getattr(diagram, "boxes", ()):
        types.extend((box.dom, box.cod))
    for ty in types:
        for atom in ty.inside:
            dim, labels = _decode_factor(atom.name, None)
            key = Ty(atom)
            if labels is not None:
                ob[key] = labels
            elif dim is not None:
                ob[key] = dim
    return ob

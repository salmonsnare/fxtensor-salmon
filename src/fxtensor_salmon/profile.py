from __future__ import annotations
import ast
from typing import Any, List, Optional, Tuple, Union


def _parse_profile(
    profile: Union[
        List[List[Union[int, List[str]]]],
        Tuple[List[int], List[int]],
    ],
) -> Tuple[List[int], List[int], Optional[Tuple[List[Any], List[Any]]]]:
    """Parse a numeric or labeled profile into dims and optional labels."""
    if isinstance(profile, tuple):
        domain_dims = list(profile[0])
        codomain_dims = list(profile[1])
        return domain_dims, codomain_dims, None

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
        return domain_dims, codomain_dims, None

    domain_labels = profile[0] if profile[0] else []
    codomain_labels = profile[1] if profile[1] else []
    labels = (domain_labels, codomain_labels)
    domain_dims = [len(dim) for dim in domain_labels]
    codomain_dims = [len(dim) for dim in codomain_labels]
    return domain_dims, codomain_dims, labels


def _constructor_profile(domain_dims, codomain_dims, labels):
    """Choose a labeled profile when labels are present, else numeric dims."""
    if labels and (labels[0] or labels[1]):
        return [labels[0], labels[1]]
    return [list(domain_dims), list(codomain_dims)]


def _parse_strand(strand_str: str) -> Any:
    """Parse a strand key as nested lists of int/str. Does not evaluate code."""
    try:
        tree = ast.parse(strand_str, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid strand format") from exc
    return _eval_strand_node(tree.body)


def _eval_strand_node(node: ast.AST) -> Any:
    if isinstance(node, ast.List):
        return [_eval_strand_node(elt) for elt in node.elts]
    if isinstance(node, ast.Constant):
        value = node.value
        if type(value) is int or type(value) is str:
            return value
    raise ValueError("Invalid strand format")

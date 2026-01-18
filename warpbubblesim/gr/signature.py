"""
Metric signature utilities for WarpBubbleSim.

Conventions:
- Default signature: (-,+,+,+) (mostly plus)
- Index ordering: (t,x,y,z) = (0,1,2,3)
"""

import numpy as np
from typing import Literal


SignatureType = Literal["mostly_plus", "mostly_minus"]


def get_signature(convention: SignatureType = "mostly_plus") -> np.ndarray:
    """
    Get the Minkowski metric for a given signature convention.

    Parameters
    ----------
    convention : str
        Either 'mostly_plus' for (-,+,+,+) or 'mostly_minus' for (+,-,-,-).

    Returns
    -------
    np.ndarray
        4x4 diagonal Minkowski metric.
    """
    if convention == "mostly_plus":
        return np.diag([-1.0, 1.0, 1.0, 1.0])
    elif convention == "mostly_minus":
        return np.diag([1.0, -1.0, -1.0, -1.0])
    else:
        raise ValueError(f"Unknown signature convention: {convention}")


def validate_metric_signature(
    metric: np.ndarray,
    expected_signature: tuple[int, ...] = (-1, 1, 1, 1)
) -> bool:
    """
    Validate that a metric has the expected signature.

    Computes eigenvalues and checks their signs.

    Parameters
    ----------
    metric : np.ndarray
        4x4 metric tensor at a single point.
    expected_signature : tuple
        Expected signs of eigenvalues, default (-1, 1, 1, 1).

    Returns
    -------
    bool
        True if signature matches, False otherwise.
    """
    eigenvalues = np.linalg.eigvalsh(metric)
    signs = np.sign(eigenvalues)
    sorted_expected = np.sort(expected_signature)
    sorted_actual = np.sort(signs)
    return np.allclose(sorted_expected, sorted_actual)


def raise_index(
    tensor_lower: np.ndarray,
    metric_inverse: np.ndarray,
    index: int = 0
) -> np.ndarray:
    """
    Raise a tensor index using the inverse metric.

    For a covariant vector v_μ, computes v^μ = g^{μν} v_ν

    Parameters
    ----------
    tensor_lower : np.ndarray
        Tensor with lower index to raise.
    metric_inverse : np.ndarray
        Inverse metric g^{μν}.
    index : int
        Which index to raise (for higher-rank tensors).

    Returns
    -------
    np.ndarray
        Tensor with raised index.
    """
    if tensor_lower.ndim == 1:
        # Vector case: v^μ = g^{μν} v_ν
        return np.einsum('mn,n->m', metric_inverse, tensor_lower)
    elif tensor_lower.ndim == 2:
        if index == 0:
            # T^μ_ν = g^{μρ} T_{ρν}
            return np.einsum('mr,rn->mn', metric_inverse, tensor_lower)
        else:
            # T_μ^ν = g^{νρ} T_{μρ}
            return np.einsum('nr,mr->mn', metric_inverse, tensor_lower)
    else:
        raise NotImplementedError("Index raising for tensors of rank > 2 not implemented")


def lower_index(
    tensor_upper: np.ndarray,
    metric: np.ndarray,
    index: int = 0
) -> np.ndarray:
    """
    Lower a tensor index using the metric.

    For a contravariant vector v^μ, computes v_μ = g_{μν} v^ν

    Parameters
    ----------
    tensor_upper : np.ndarray
        Tensor with upper index to lower.
    metric : np.ndarray
        Metric tensor g_{μν}.
    index : int
        Which index to lower (for higher-rank tensors).

    Returns
    -------
    np.ndarray
        Tensor with lowered index.
    """
    if tensor_upper.ndim == 1:
        # Vector case: v_μ = g_{μν} v^ν
        return np.einsum('mn,n->m', metric, tensor_upper)
    elif tensor_upper.ndim == 2:
        if index == 0:
            # T_{μν} from T^μ_ν: T_{μν} = g_{μρ} T^ρ_ν
            return np.einsum('mr,rn->mn', metric, tensor_upper)
        else:
            # T_{μν} from T_μ^ν: T_{μν} = g_{νρ} T_μ^ρ
            return np.einsum('nr,mr->mn', metric, tensor_upper)
    else:
        raise NotImplementedError("Index lowering for tensors of rank > 2 not implemented")


def contract_indices(
    tensor: np.ndarray,
    metric: np.ndarray,
    indices: tuple[int, int]
) -> np.ndarray:
    """
    Contract two indices of a tensor using the metric.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to contract.
    metric : np.ndarray
        Metric tensor.
    indices : tuple
        Pair of indices to contract.

    Returns
    -------
    np.ndarray
        Contracted tensor.
    """
    # This is a simplified implementation for common cases
    i, j = indices
    if tensor.ndim == 2 and i == 0 and j == 1:
        # Trace: T^μ_μ or g^{μν} T_{μν}
        return np.einsum('ij,ij->', metric, tensor)
    else:
        raise NotImplementedError("General index contraction not implemented")


def compute_determinant(metric: np.ndarray) -> float:
    """
    Compute the determinant of the metric tensor.

    Parameters
    ----------
    metric : np.ndarray
        4x4 metric tensor.

    Returns
    -------
    float
        Determinant of the metric.
    """
    return np.linalg.det(metric)


def compute_volume_element(metric: np.ndarray) -> float:
    """
    Compute the invariant volume element sqrt(-det(g)).

    For signature (-,+,+,+), det(g) < 0, so we use sqrt(-det(g)).

    Parameters
    ----------
    metric : np.ndarray
        4x4 metric tensor.

    Returns
    -------
    float
        Volume element sqrt(-det(g)).
    """
    det = compute_determinant(metric)
    return np.sqrt(np.abs(det))

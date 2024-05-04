"""Methods to convert complex-valued arrays into their real-valued counterparts.

Vector: v = v_R + i v_I --> [ v_R, v_I ]

Matrix: M = M_R + i M_I --> ⌈ M_R, -M_I ⌉
                            ⌊ M_I,  M_R ⌋

So that: M v = (M_R v_R - M_I v_I) + i (M_I v_R + M_R v_I)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as np

if TYPE_CHECKING:
    from jax import Array


def complex_to_real_vec(array: Array) -> Array:
    """Convert at complex-valued vector its real-valued counterpart."""
    return np.concatenate([array.real, array.imag])


def complex_to_real_mat(array: Array) -> Array:
    """Convert at complex-valued matrix its real-valued counterpart."""
    return np.block([[array.real, -array.imag], [array.imag, array.real]])


def real_to_complex_vec(array: Array) -> Array:
    """Invert complex_to_real_vec."""
    dim = array.shape[0] // 2
    return array[:dim] + 1j * array[dim:]


def real_to_complex_mat(array: Array) -> Array:
    """Invert complex_to_real_mat."""
    dim = array.shape[0] // 2
    return array[:dim, :dim] + 1j * array[dim:, :dim]

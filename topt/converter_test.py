"""Unit tests for converter.py."""

import jax

import topt


def test_conversions(dim: int = 5) -> None:
    """Test conversions with random numbers."""
    key = jax.random.PRNGKey(0)
    vec = jax.random.normal(key, (dim,), dtype=complex)
    mat = jax.random.normal(key, (dim, dim), dtype=complex)

    vec_real = topt.converter.complex_to_real_vec(vec)
    mat_real = topt.converter.complex_to_real_mat(mat)

    assert jax.numpy.array_equal(vec, topt.converter.real_to_complex_vec(vec_real))
    assert jax.numpy.array_equal(mat, topt.converter.real_to_complex_mat(mat_real))
    assert jax.numpy.allclose(mat @ vec, topt.converter.real_to_complex_vec(mat_real @ vec_real))

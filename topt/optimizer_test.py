"""Unit tests for optimizer.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

import topt

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike


def test_time_step(dim: int = 5, time_step: float = 1e-3) -> None:
    """Accuracy of the dynamical constraint for a single time step."""
    key = jax.random.PRNGKey(0)
    state_init = jax.random.normal(key, (dim,), dtype=complex)
    generator = jax.random.normal(key, (dim, dim), dtype=complex)

    state_step = jax.scipy.linalg.expm(time_step * generator) @ state_init
    constraint = topt.optimizer.get_time_step_constraint(
        state_init, state_step, generator, time_step
    )
    assert jax.numpy.abs(constraint.real).max() < time_step**2
    assert jax.numpy.abs(constraint.imag).max() < time_step**2


def test_dynamical_constraints(
    dim: int = 4,
    num_time_steps: int = 10,
    num_dynamic_params: int = 2,
    num_static_params: int = 2,
) -> None:
    """Dynamical constraint (and its Jacobian) for a full trajectory."""
    time_span = 1
    time_step = time_span / num_time_steps

    # construct a random generator

    key = jax.random.PRNGKey(0)
    mat_time = jax.random.normal(key, (dim, dim))
    mats_dynamic = [jax.random.normal(key, (dim, dim)) for _ in range(num_dynamic_params)]
    mats_static = [jax.random.normal(key, (dim, dim)) for _ in range(num_static_params)]

    def generator(time: ArrayLike, dynamic_params: Array, static_params: Array) -> Array:
        return (
            time * mat_time
            + sum(dynamic_params[ii] * mats_dynamic[ii] for ii in range(num_dynamic_params))
            + sum(static_params[ii] * mats_static[ii] for ii in range(num_static_params))
        )

    # construct a random trajectory

    initial_state = jax.random.normal(key, (dim,))
    initial_params = jax.random.normal(key, (num_dynamic_params + num_static_params,))
    trajectory = topt.optimizer.build_trajectory(
        initial_state,
        initial_params,
        generator,
        time_span,
        num_time_steps,
        num_static_params,
        rtol=time_step**4,
        atol=time_step**4,
    )

    # check that constraints are satisfied

    constraint_funcs = topt.optimizer.get_dynamical_constraints(
        initial_state,
        generator,
        num_time_steps,
        num_dynamic_params,
        num_static_params,
    )
    constraint = constraint_funcs["fun"](trajectory)
    assert jax.numpy.abs(constraint).max() < time_step**2

    # check derivatives of the constraint function

    constraint_jac = constraint_funcs["jac"](trajectory)
    constraint_jac_AD = jax.jacfwd(constraint_funcs["fun"])(trajectory)
    assert jax.numpy.allclose(constraint_jac, constraint_jac_AD)
    assert not jax.numpy.any(constraint_funcs["hess"](trajectory))

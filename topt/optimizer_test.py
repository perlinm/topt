"""Unit tests for optimizer.py."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax

import topt

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike


def get_random_generator(
    key: Array, dim: int, num_dynamic_params: int, num_static_params: int
) -> Callable[[ArrayLike, Array, Array], Array]:
    """Construct a random generator."""
    mat_time = jax.random.normal(key, (dim, dim))
    mats_dynamic = [jax.random.normal(key, (dim, dim)) for _ in range(num_dynamic_params)]
    mats_static = [jax.random.normal(key, (dim, dim)) for _ in range(num_static_params)]

    def generator(
        time: ArrayLike = jax.numpy.array([1]),
        dynamic_params: Array = jax.numpy.array([]),
        static_params: Array = jax.numpy.array([]),
    ) -> Array:
        return jax.numpy.array(
            time * mat_time
            + sum(dynamic_params[ii] * mats_dynamic[ii] for ii in range(num_dynamic_params))
            + sum(static_params[ii] * mats_static[ii] for ii in range(num_static_params))
        )

    return generator


def test_time_step(
    dim: int = 5,
    num_dynamic_params: int = 2,
    num_static_params: int = 2,
    time_step: float = 1e-3,
    time_span: float = 1.0,
    time: float = 0.5,
    seed: int = 0,
) -> None:
    """Dynamical constraint for a sigle time step."""
    key = jax.random.PRNGKey(seed)

    state_init = jax.random.normal(key, (dim,))
    dynamic_params = jax.random.normal(key, (num_dynamic_params,))
    static_params = jax.random.normal(key, (num_static_params,))
    generator = get_random_generator(key, dim, num_dynamic_params, num_static_params)

    # test accuracy of base constraint

    gen_mat = generator(time, dynamic_params, static_params)
    state_step = jax.scipy.linalg.expm(time_step * gen_mat) @ state_init
    constraint = topt.optimizer.get_time_step_constraint(state_init, state_step, gen_mat, time_step)
    assert jax.numpy.abs(constraint.real).max() < time_step**2
    assert jax.numpy.abs(constraint.imag).max() < time_step**2

    # test correctness of jacobian

    generator_jac = topt.optimizer.get_generator_jac(generator, num_static_params)
    gen_jac_mat = generator_jac(time, dynamic_params, static_params)
    jacobian = topt.optimizer.get_time_step_jacobian_block(
        state_init, state_step, gen_mat, gen_jac_mat, time_step, time_span, time
    )

    def constraint_array_func(data: Array) -> Array:
        state_init = data[:dim]
        state_step = data[dim : 2 * dim]
        dynamic_params = data[2 * dim : 2 * dim + num_dynamic_params]
        static_params = data[2 * dim + num_dynamic_params : -1]
        time_scale = data[-1] / time_span
        gen_mat = generator(time * time_scale, dynamic_params, static_params)
        return topt.optimizer.get_time_step_constraint(
            state_init, state_step, gen_mat, time_step * time_scale
        )

    data = jax.numpy.concatenate(
        [state_init, state_step, dynamic_params, static_params, jax.numpy.array([time_span])]
    )
    jacobian_with_autodiff = jax.jacrev(constraint_array_func)(data)
    assert jax.numpy.allclose(jacobian, jacobian_with_autodiff)


def test_dynamical_constraints(
    dim: int = 4,
    time_span: float = 1.0,
    num_time_steps: int = 10,
    num_dynamic_params: int = 2,
    num_static_params: int = 2,
    seed: int = 0,
) -> None:
    """Dynamical constraint (and its Jacobian) for a full trajectory."""
    key = jax.random.PRNGKey(seed)
    time_step = time_span / num_time_steps

    # construct a random trajectory

    initial_state = jax.random.normal(key, (dim,))
    initial_params = jax.random.normal(key, (num_dynamic_params + num_static_params,))
    generator = get_random_generator(key, dim, num_dynamic_params, num_static_params)
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
    constraint = constraint_funcs["fun"](trajectory)  # pylint: disable=not-callable
    assert jax.numpy.abs(constraint).max() < time_step**2

    # check derivatives of the constraint function

    constraint_jac = constraint_funcs["jac"](trajectory)  # pylint: disable=not-callable
    constraint_jac_with_autodiff = jax.jacfwd(constraint_funcs["fun"])(trajectory)
    assert jax.numpy.allclose(constraint_jac, constraint_jac_with_autodiff)
    assert not jax.numpy.any(constraint_funcs["hess"](trajectory))

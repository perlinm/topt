"""Unit tests for optimizer.py."""

from __future__ import annotations

import collections
import functools
import random
import unittest.mock
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
    dim: int = 3,
    num_dynamic_params: int = 2,
    num_static_params: int = 2,
    time_step: float = 1e-3,
    seed: int = 0,
) -> None:
    """Dynamical constraint for a sigle time step."""
    key = jax.random.PRNGKey(seed)
    time = random.random()
    time_span = time + random.random()

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
    jacobian_autodiff = jax.jacrev(constraint_array_func)(data)
    assert jax.numpy.allclose(jacobian, jacobian_autodiff)


def test_dynamical_constraints(
    dim: int = 3,
    time_span: float = 1.0,
    num_time_steps: int = 10,
    num_dynamic_params: int = 2,
    num_static_params: int = 2,
    seed: int = 1,
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

    con_funcs = topt.optimizer.get_dynamical_constraints(
        initial_state,
        generator,
        num_time_steps,
        num_dynamic_params,
        num_static_params,
    )
    constraint = con_funcs["fun"](trajectory)  # pylint: disable=not-callable
    jacobian = con_funcs["jac"](trajectory)  # pylint: disable=not-callable
    hessian: Array = con_funcs["hess"](trajectory)  # pylint: disable=not-callable
    assert jax.numpy.abs(constraint).max() < time_step**2

    # check that second derivatives with respect to state variables are zero

    assert not jax.numpy.any(hessian[:, : dim * num_time_steps, : dim * num_time_steps])

    # check correctness of the derivatives of the constraint function

    def split_constraint_func(
        states: Array, dynamic_params: Array, static_params: Array, time_span: Array
    ) -> Array:
        trajectory = jax.numpy.concatenate(
            [
                states.ravel(),
                dynamic_params.ravel(),
                static_params,
                jax.numpy.array(time_span, ndmin=1),
            ]
        )
        return con_funcs["fun"](trajectory)  # pylint: disable=not-callable

    splitter = functools.partial(
        topt.optimizer.split_trajectory,
        num_time_steps=num_time_steps,
        state_dim=initial_state.size,
        num_static_params=num_static_params,
    )
    con_funcs_autodiff = topt.optimizer.get_derivatives(split_constraint_func, splitter)
    jacobian_autodiff = con_funcs_autodiff["jac"](trajectory)  # pylint: disable=not-callable
    hessian_autodiff = con_funcs_autodiff["hess"](trajectory)  # pylint: disable=not-callable
    assert jax.numpy.allclose(jacobian, jacobian_autodiff)
    assert jax.numpy.allclose(hessian, hessian_autodiff)


def test_optimize_trajectory_call(
    dim: int = 5,
    num_time_steps: int = 10,
    num_dynamic_params: int = 2,
    num_static_params: int = 2,
    seed: int = 2,
) -> None:
    """Mock optimizing a trajectory to ensure that nothing raises an error."""
    key = jax.random.PRNGKey(seed)

    initial_state = jax.random.normal(key, (dim,))
    initial_params = jax.random.normal(key, (num_dynamic_params + num_static_params,))
    generator = get_random_generator(key, dim, num_dynamic_params, num_static_params)
    time_span = random.random()

    trajectory = topt.optimizer.build_trajectory(
        initial_state,
        initial_params,
        generator,
        time_span,
        num_time_steps,
        num_static_params,
    )
    states = trajectory[: dim * num_time_steps].reshape(num_time_steps, dim)
    dynamic_params = trajectory[states.size : -1 - num_static_params].reshape(num_time_steps, -1)
    static_params = trajectory[-1 - num_static_params : -1]

    class Problem:
        """Mock class for cyipopt.Problem."""

        def __init__(self, **kwargs: object) -> None: ...

        def add_option(self, *args: object) -> None:
            """Placeholder method."""

        def solve(self, *args: object) -> tuple[Array, dict[str, int]]:
            """Return the initial trajectory."""
            return trajectory, collections.defaultdict(int)

    with unittest.mock.patch("cyipopt.Problem", Problem):
        result = topt.optimizer.optimize_trajectory(
            initial_state,
            initial_params,
            generator,
            time_span,
            num_time_steps,
            objective=lambda states, dynamic_params, static_params, time_span: jax.numpy.array([]),
            num_static_params=num_static_params,
        )
        assert jax.numpy.array_equal(result.x.states[0, :], initial_state)
        assert jax.numpy.array_equal(result.x.states[1:, :], states)
        assert jax.numpy.array_equal(result.x.dynamic_params, dynamic_params)
        assert jax.numpy.array_equal(result.x.static_params, static_params)
        assert jax.numpy.array_equal(result.x.time_span, jax.numpy.array(time_span))

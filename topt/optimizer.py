"""Methods to optimize trajectories using IPOPT and Pade approximants.

This module shamelessly borrows ideas from https://github.com/aarontrowbridge/Pico.jl
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import cyipopt
import jax
import jax.numpy as np
import scipy
from jax import Array
from jax.typing import ArrayLike


@dataclasses.dataclass
class Trajectory:
    """Container for trajectory data."""

    states: Array
    dynamic_params: Array
    static_params: Array
    time_span: Array


SplitTrajectoryFunc = Callable[[Array, Array, Array, Array], Array]
SplitParamFunc = Callable[[ArrayLike, Array, Array], Array]


def optimize_trajectory(
    initial_state: Array,
    initial_params: Array,
    generator: Callable[[ArrayLike, Array, Array], Array],
    time_span: float,
    num_time_steps: int,
    objective: SplitTrajectoryFunc,
    *constraints: SplitTrajectoryFunc | tuple[SplitTrajectoryFunc, ArrayLike, ArrayLike],
    num_static_params: int = 0,
    objective_grad: SplitTrajectoryFunc | None = None,
    objective_hess: SplitTrajectoryFunc | None = None,
    generator_jac: SplitParamFunc | None = None,
    time_bounds: tuple[float, float] | None = None,
    tol: float = 1e-4,  # error tolerance passed to IPOPT
) -> scipy.optimize.OptimizeResult:
    """Optimize a trajectory with direct collocation.

    This method represets a trajectory by discretizing time, and storing a vector (state, control
    parameters, etc.) at each time point.  Optimization constraints enforce that neighboring
    vectors in time satisfy linear equations of motion.  These dynamical constraints, together with
    a provided objective function and possibly additional constraints on the trajectory, define an
    optimization problem that this method solves with IPOPT.

    Args:
        initial_state: the initial state that is time-evolved with equations of motion.
        initial_params: initial parameters for constructing a trajectory from the initial_state.
        generator: a function that maps (time, dynamic_params, static_params) onto a matrix that
            generates time evolution.
        time_span: the total time for which to evolve, either as a fixed value or a tuple that
            specifies (initial_value, lower_bound, upper_bound).
        num_time_steps: number of time steps with which to discretize the trajectory.
        objective: an objective function (to be minimized) of the entire trajectory.
        constraints (optional): additional constraints to enforce for the trajectory.  Each
            constraint may be specified as a function that must equal zero, or as a tuple that
            specifies (function, lower_bounds, upper_bounds).
        num_static_params: the number of static (not-time-varying) parameters.
        objective_grad: the gradient of the objective function.
        objective_hess: the hessian of the objective function.
        generator_jac: the jacobian of the generator of time evolution.
        time_bounds: lower and upper bounds on the allowed time span.  If bounds are not
            provided, the time span is constrained to a fixed value.
        tol: an error tolerance passed to IPOPT.

    The objective and constraint functions must both accept four arrays corresponding to
        (state, dynamic_params, static_params, time_span).
    These functions must likewise return an array.
    """
    num_dynamic_params = initial_params.size - num_static_params
    time_lb, time_ub = time_bounds if time_bounds is not None else (time_span, time_span)

    # build an initial trajectory that satisfies equations of motion
    initial_trajectory = build_trajectory(
        initial_state, initial_params, generator, time_span, num_time_steps, num_static_params
    )

    # collect objective function, constraint functions, and derivatives thereof
    splitter = functools.partial(
        split_trajectory,
        num_time_steps=num_time_steps,
        state_dim=initial_state.size,
        num_static_params=num_static_params,
    )
    _objective = get_derivatives(objective, splitter, jac=objective_grad, hess=objective_hess)
    _constraints = [
        get_derivatives(con if callable(con) else con[0], splitter) for con in constraints
    ] + [
        get_dynamical_constraints(
            initial_state,
            generator,
            num_time_steps,
            num_dynamic_params,
            num_static_params,
            generator_jac=generator_jac,
        )
    ]

    # Set lower/upper bounds for decision variables.  Only time should be bounded here.
    var_lb = np.concatenate([-np.inf * np.ones(initial_trajectory.size - 1), np.array([time_lb])])
    var_up = np.concatenate([+np.inf * np.ones(initial_trajectory.size - 1), np.array([time_ub])])

    # identify constraint dimensions and lower/upper bounds
    con_dims = cyipopt.get_constraint_dimensions(_constraints, initial_trajectory)
    con_lower_bounds = [
        np.zeros(dim) if callable(con) else np.array(con[1], ndmin=1)
        for con, dim in zip(constraints, con_dims)
    ]
    con_upper_bounds = [
        np.zeros(dim) if callable(con) else np.array(con[2], ndmin=1)
        for con, dim in zip(constraints, con_dims)
    ]
    con_lb = np.concatenate(con_lower_bounds + [np.zeros(con_dims[-1])])
    con_up = np.concatenate(con_upper_bounds + [np.zeros(con_dims[-1])])

    # identify jacobian sparsity, necessary for compatibility with cyipopt.IpoptProblemWrapper
    sparse_jacs, jac_nnz_row, jac_nnz_col = cyipopt.scipy_interface._get_sparse_jacobian_structure(
        _constraints, initial_trajectory
    )

    # build intermediate problem object
    problem_object = cyipopt.IpoptProblemWrapper(
        _objective["fun"],
        jac=_objective["jac"],
        hess=_objective["hess"],
        constraints=_constraints,
        con_dims=con_dims,
        sparse_jacs=sparse_jacs,
        jac_nnz_row=jac_nnz_row,
        jac_nnz_col=jac_nnz_col,
    )

    # build full cyipopt problem object
    problem = cyipopt.Problem(
        n=initial_trajectory.size,
        m=con_lb.size,
        problem_obj=problem_object,
        lb=var_lb,
        ub=var_up,
        cl=con_lb,
        cu=con_up,
    )
    problem.add_option("mu_strategy", "adaptive")
    problem.add_option("tol", tol)

    # solve the problem!
    trajectory_data, info = problem.solve(initial_trajectory)

    # prepend initial state and return
    trajectory = Trajectory(*splitter(trajectory_data))
    trajectory.states = np.vstack([initial_state, trajectory.states])
    return scipy.optimize.OptimizeResult(
        x=trajectory,
        success=info["status"] == 0,
        status=info["status"],
        message=info["status_msg"],
        fun=info["obj_val"],
        info=info,
        nfev=problem_object.nfev,
        njev=problem_object.njev,
        nit=problem_object.nit,
    )


def build_trajectory(
    initial_state: Array,
    params: Array,
    generator: Callable[[ArrayLike, Array, Array], Array],
    time_span: ArrayLike,
    num_time_steps: int,
    num_static_params: int,
    *,
    rtol: float = 1e-3,  # default for scipy.integrate.solve_ivp
    atol: float = 1e-6,  # default for scipy.integrate.solve_ivp
) -> Array:
    """Build the trajectory of a state satisfying equations of motion."""
    num_dynamic_params = params.size - num_static_params
    dynamic_params = params[:num_dynamic_params]
    static_params = params[num_dynamic_params:]

    times = np.linspace(0, time_span, num_time_steps + 1)[1:]
    result = scipy.integrate.solve_ivp(
        lambda time, state: generator(time, dynamic_params, static_params) @ state,
        (0, times[-1]),
        initial_state,
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    states = result.y.T

    dynamic_params = np.broadcast_to(dynamic_params, (num_time_steps, num_dynamic_params))
    time_span = np.array([time_span])
    return np.concatenate([states.ravel(), dynamic_params.ravel(), static_params, time_span])


def split_trajectory(
    trajectory: Array, num_time_steps: int, state_dim: int, num_static_params: int
) -> tuple[Array, Array, Array, Array]:
    """Split trajectory data into (states, dynamic_params, static_params, time_span)."""
    states = trajectory[: num_time_steps * state_dim]
    dynamic_params = trajectory[num_time_steps * state_dim : -num_static_params - 1]
    static_params = trajectory[-num_static_params - 1 : -1]
    time_span = trajectory[-1]
    return (
        states.reshape(num_time_steps, state_dim),
        dynamic_params.reshape(num_time_steps, -1),
        static_params,
        time_span,
    )


def get_derivatives(
    function: SplitTrajectoryFunc,
    splitter: Callable[[Array], tuple[Array, Array, Array, Array]],
    *,
    jac: SplitTrajectoryFunc | None = None,
    hess: SplitTrajectoryFunc | None = None,
) -> dict[str, Callable[[Array], Array]]:
    """Compile a function of all trajectory data, as well as its derivatives."""

    def with_full_trajectory(function: SplitTrajectoryFunc) -> Callable[[Array], Array]:
        return lambda trajectory: function(*splitter(trajectory))

    _function = with_full_trajectory(function)
    _function_jac = with_full_trajectory(jac) if jac is not None else jax.jacrev(_function)
    _function_hess = (
        with_full_trajectory(hess)
        if hess is not None
        else jax.jacrev(_function_jac)  # type:ignore[arg-type]
    )
    return {
        "fun": jax.jit(_function),
        "jac": jax.jit(_function_jac),  # type:ignore[arg-type]
        "hess": jax.jit(_function_hess),  # type:ignore[arg-type]
    }


def get_dynamical_constraints(
    initial_state: Array,
    generator: Callable[[ArrayLike, Array, Array], Array],
    num_time_steps: int,
    num_dynamic_params: int,
    num_static_params: int,
    generator_jac: SplitParamFunc | None = None,
) -> dict[str, Callable[[Array], Array]]:
    """Build constraints (and derivatives thereof) induced by equations of motion."""
    state_dim = initial_state.size
    states_size = state_dim * num_time_steps
    dynamic_params_size = num_dynamic_params * num_time_steps
    trajectory_size = states_size + dynamic_params_size + num_static_params + 1

    # function to split a trajectory into its components
    splitter = functools.partial(
        split_trajectory,
        num_time_steps=num_time_steps,
        state_dim=state_dim,
        num_static_params=num_static_params,
    )

    # construct dynamical constraints: neighboring states (in time) satisfy equations of motion

    def constraints(trajectory: Array) -> Array:
        states, dynamic_params, static_params, time_span = splitter(trajectory)
        time_step = time_span / num_time_steps

        # constraint for first time step
        constraints_init = get_time_step_constraint(
            initial_state,
            states[0, :],
            generator(time_step / 2, dynamic_params[0, :], static_params),
            time_step,
        )

        # constraint for remaining time steps
        constraints_bulk = [
            get_time_step_constraint(
                states[ss - 1, :],
                states[ss, :],
                generator(time_step * (ss + 0.5), dynamic_params[ss, :], static_params),
                time_step,
            )
            for ss in range(1, num_time_steps)
        ]

        return np.concatenate([constraints_init, np.array(constraints_bulk).ravel()])

    # compute the jacobian of the generator, if necessary
    if generator_jac is None:
        generator_jac = get_generator_jac(generator, num_static_params)

    # identify columns of the constraint jacobian that are occupied by data at a certain time index

    def get_jacobian_cols(time_index: int) -> Array:
        state_cols = np.arange(state_dim * (time_index - 1), state_dim * (time_index + 1))
        dynamic_param_cols = np.arange(
            states_size + num_dynamic_params * time_index,
            states_size + num_dynamic_params * (time_index + 1),
        )
        last_cols = np.arange(states_size + dynamic_params_size, trajectory_size)
        return np.concatenate([state_cols, dynamic_param_cols, last_cols])

    # Build the jacobian of the dynamical constraint function.  This jacobian can be obtained with
    # automatic differentiation, but building it manually makes calls to the jacobian much more
    # efficient.

    def jacobian(trajectory: Array) -> Array:
        states, dynamic_params, static_params, time_span = splitter(trajectory)
        time_step = time_span / num_time_steps

        zeros = np.zeros((state_dim, trajectory.size))
        blocks = []

        # first time step
        time = time_step / 2
        jacobian_block = get_time_step_jacobian_block(
            initial_state,
            states[0, :],
            generator(time, dynamic_params[0, :], static_params),
            generator_jac(time, dynamic_params[0, :], static_params),
            time_step,
            time_span,
            time,
        )
        cols = get_jacobian_cols(0)
        mat = zeros.at[:, cols[state_dim:]].set(jacobian_block[:, state_dim:])
        blocks.append(mat)

        # remaining time steps
        for ss in range(1, num_time_steps):
            time = time_step * (ss + 0.5)
            jacobian_block = get_time_step_jacobian_block(
                states[ss - 1, :],
                states[ss, :],
                generator(time, dynamic_params[ss, :], static_params),
                generator_jac(time, dynamic_params[ss, :], static_params),
                time_step,
                time_span,
                time,
            )
            cols = get_jacobian_cols(ss)
            mat = zeros.at[:, cols].set(jacobian_block)
            blocks.append(mat)

        return np.vstack(blocks)

    return {
        "fun": jax.jit(constraints),
        "jac": jax.jit(jacobian),
        "hess": jax.jit(jax.jacrev(jacobian)),
    }


def get_generator_jac(generator: SplitParamFunc, num_static_params: int) -> SplitParamFunc:
    """The Jacobian of a generator function."""

    def flat_generator(data: Array) -> Array:
        time = data[0]
        dynamic_params = data[1:] if not num_static_params else data[1:-num_static_params]
        static_params = data[-num_static_params:]
        return generator(time, dynamic_params, static_params).ravel()

    flat_generator_jac = jax.jacfwd(flat_generator)

    def generator_jac(time: ArrayLike, dynamic_params: Array, static_params: Array) -> Array:
        time = np.array(time, ndmin=1)
        return flat_generator_jac(np.concatenate([time, dynamic_params, static_params]))

    return generator_jac


def get_time_step_constraint(
    state_init: Array, state_step: Array, generator: Array, time_step: ArrayLike
) -> Array:
    """Get the constraint enforcing time evolution by a single time step.

    Time evolution is nominally disretized with
        state_step = exp(time_step * generator) @ state_init.

    The exponential exp(step_generator) is then further appoximated by a (2, 2) Padé approximant.
    See: https://en.wikipedia.org/wiki/Pad%C3%A9_table#An_example_%E2%80%93_the_exponential_function
    """
    c_1, c_2 = time_step / 2.0, time_step**2 / 12.0  # coefficients for a (2, 2) Padé approximant
    state_p = state_init + state_step
    state_m = state_init - state_step
    return state_m + c_1 * (generator @ state_p) + c_2 * (generator @ (generator @ state_m))


def get_time_step_jacobian_block(
    state_init: Array,
    state_step: Array,
    generator: Array,
    generator_jac: Array,
    time_step: ArrayLike,
    time_span: ArrayLike,
    time: ArrayLike,
) -> Array:
    """Block of the Jacobian of get_time_step_constraint."""
    time_ratio = time / time_span
    step_gen = time_step * generator

    # derivative with respect to states
    iden = np.identity(generator.shape[0])
    mat_o = step_gen / 2.0
    mat_e = iden + step_gen @ step_gen / 12.0
    state_block = np.hstack([mat_o + mat_e, mat_o - mat_e])

    c_1, c_2 = time_step / 2.0, time_step**2 / 12.0  # coefficients for a (2, 2) Padé approximant
    state_p = state_init + state_step
    state_m = state_init - state_step

    # derivative with respect to generator arguments
    dG_dT = np.moveaxis(generator_jac.reshape((state_init.size, state_init.size, -1)), 2, 0)
    term_1 = dG_dT @ state_p
    term_2 = dG_dT @ (generator @ state_m) + (generator @ (dG_dT @ state_m).T).T
    param_block = c_1 * term_1 + c_2 * term_2

    # derivative with respect to the time_span
    term_1 = generator @ state_p
    term_2 = generator @ (generator @ state_m)
    term_3 = param_block[0, :]
    time_block = c_1 / time_span * term_1 + c_2 * 2.0 / time_span * term_2 + time_ratio * term_3

    return np.hstack([state_block, param_block[1:, :].T, time_block.reshape(-1, 1)])

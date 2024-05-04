#!/usr/bin/env python3
"""Demonstration: learning a Rabi phase profile for a pi pulse in the presence of detuning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as np
import matplotlib.pyplot as plt

import topt

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike


# enable 64 bit floating point precision
jax.config.update("jax_enable_x64", True)

# use the CPU instead of GPU and mute all warnings if no GPU/TPU is found
jax.config.update("jax_platform_name", "cpu")


def generator(time: ArrayLike, dynamic_params: Array, static_params: Array) -> Array:
    """Generator for constant axial (Z) field and dynamic transverse (XY) field."""
    op_I = np.identity(2, dtype=complex)
    op_z = np.array([[1, 0], [0, -1]], dtype=complex)
    op_x = np.array([[0, 1], [1, 0]], dtype=complex)
    op_y = -1j * op_z @ op_x

    gen_I = topt.converter.complex_to_real_mat(-1j * np.pi * op_I)
    gen_x = topt.converter.complex_to_real_mat(-1j * np.pi * op_x)
    gen_y = topt.converter.complex_to_real_mat(-1j * np.pi * op_y)
    gen_z = topt.converter.complex_to_real_mat(-1j * np.pi * op_z)
    return (
        gen_z
        + dynamic_params[0] * gen_x
        + dynamic_params[1] * gen_y
        + (static_params[0] + time) * gen_I  # included to test the jacobian of this method
    )


def objective(
    states: Array, dynamic_params: Array, static_params: Array, time_span: Array
) -> Array:
    """Objective to minimize: population of the |0> state."""
    last_state = topt.converter.real_to_complex_vec(states[-1, :])
    pop_0 = np.abs(last_state[0]) ** 2
    return pop_0


def field_constraint(
    states: Array, dynamic_params: Array, static_params: Array, time_span: Array
) -> Array:
    """Constraint on the magnitude of the transverse (XY) field."""
    return np.linalg.norm(dynamic_params, ord=2, axis=1)


num_time_steps = 100
time_init = 0.2
time_bounds = (0, 1)

initial_state = np.array([1, 0], dtype=complex)
initial_params = np.array([1, 0, 0], dtype=float)
# initial_params = np.array([], dtype=float)

num_static_params = 1


result = topt.optimizer.optimize_trajectory(
    topt.converter.complex_to_real_vec(initial_state),
    initial_params,
    generator,
    time_init,
    num_time_steps,
    objective,
    (field_constraint, np.ones(num_time_steps), np.ones(num_time_steps)),
    num_static_params=num_static_params,
    time_span_bounds=time_bounds,
)
states = result.x.states
dynamic_params = result.x.dynamic_params
static_params = result.x.static_params
time_span = result.x.time_span

print()
print(np.abs(topt.converter.real_to_complex_vec(states[-1, :])) ** 2)
print(static_params)
print(time_span)
print()

times = np.linspace(0, time_span, num_time_steps + 1)

op_z = np.array([[1, 0], [0, -1]], dtype=complex)
op_x = np.array([[0, 1], [1, 0]], dtype=complex)
op_y = -1j * op_z @ op_x

op_x = topt.converter.complex_to_real_mat(op_x)
op_y = topt.converter.complex_to_real_mat(op_y)
op_z = topt.converter.complex_to_real_mat(op_z)

x_vals = np.array([(state @ op_x @ state).real for state in states])
y_vals = np.array([(state @ op_y @ state).real for state in states])
z_vals = np.array([(state @ op_z @ state).real for state in states])

plt.plot(times, x_vals, label="x")
plt.plot(times, y_vals, label="y")
plt.plot(times, z_vals, label="z")
plt.plot(times, x_vals**2 + y_vals**2 + z_vals**2, label="norm")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

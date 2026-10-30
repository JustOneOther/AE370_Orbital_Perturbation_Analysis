from scipy.optimize import fsolve

import numpy as np
import orbits_alt as orbits


def trapezoid_method(f, x_0, dt):
	return lambda x_1: x_0 + (f(x_0) + f(x_1)) * dt / 2 - x_1


class Integrator:
	"""
	Shared setup: state history, gravitational parameters, and the right-hand side
	"""
	def __init__(self, orbiters: list[orbits.Orbiter], planets: list[orbits.Planet], step_size: float):
		state_0, self.grav_param_mat = orbits.build_state(orbiters, planets)
		self.state = [state_0]

		self.n_orbiters = len(orbiters)
		self.step_size = step_size

		self.t = 0
		self._state_array = None

	def _f(self, state):
		return orbits.f_with_n_orbiters(state, self.grav_param_mat, self.n_orbiters)

	def _append(self, x_1):
		self.state.append(x_1)
		self.t += self.step_size
		self._state_array = None  # Rebuild on next access so it includes this step

	@property
	def state_array(self):
		if self._state_array is None:
			self._state_array = np.array(self.state)

		return self._state_array


class BDF2(Integrator):
	def __init__(self, orbiters: list[orbits.Orbiter], planets: list[orbits.Planet], step_size: float):
		super().__init__(orbiters, planets, step_size)

		self._first_step(trapezoid_method(
			self._f,
			self.state[0],
			self.step_size))

	def _first_step(self, f):
		"""Calculate first step with specified lambda to fsolve"""

		self._append(fsolve(f, self.state[0]))

	def step(self):
		x_1 = fsolve(
			lambda att: self.state[-2] - 4 * self.state[-1] + 3 * att - 2 * self.step_size * self._f(att),
			self.state[-1]
		)
		self._append(x_1)


class RK4(Integrator):
	def step(self):
		x_0 = self.state[-1]
		dt = self.step_size

		k_1 = self._f(x_0)
		k_2 = self._f(x_0 + dt / 2 * k_1)
		k_3 = self._f(x_0 + dt / 2 * k_2)
		k_4 = self._f(x_0 + dt * k_3)

		self._append(x_0 + dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4))

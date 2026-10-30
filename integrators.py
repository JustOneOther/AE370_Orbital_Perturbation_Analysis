from scipy.optimize import fsolve

import numpy as np
import orbits


def trapezoid_method(f, x_0, dt):
	return lambda x_1: x_0 + (f(x_0) + f(x_1)) * dt / 2 - x_1


class BDF2:
	def __init__(self, orbiters: list[orbits.Orbiter], planets: list[orbits.Planet], step_size: float):
		orbiters.sort(key=lambda x: x.id)
		planets.sort(key=lambda x: x.id)

		bodies = orbiters + planets
		self.state = [np.zeros(6 * len(bodies))]
		for n, bod in enumerate(bodies):
			self.state[0][n * 6:(n + 1) * 6] = [*bod.position, *bod.velocity]

		self.grav_param_mat = np.array([i.grav_param for i in planets])
		self.n_orbiters = len(orbiters)
		self.step_size = step_size

		self.t = 0
		self._first_step(trapezoid_method(
			self._f,
			self.state[0],
			self.step_size))

		self._state_array = None

	def _f(self, state):
		return orbits.f_with_n_orbiters(state, self.grav_param_mat, self.n_orbiters)

	def _first_step(self, f):
		"""Calculate first step with specified lambda to fsolve"""

		self.state.append(fsolve(f, self.state[0]))
		self.t += self.step_size

	def step(self):
		x_1 = fsolve(
			lambda att: self.state[-2] - 4 * self.state[-1] + 3 * att - 2 * self.step_size * self._f(att),
			self.state[-1]
		)
		self.state.append(x_1)
		self.t += self.step_size

	@property
	def state_array(self):
		if self._state_array is None:
			self._state_array = np.array(self.state)

		return self._state_array

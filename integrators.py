import matplotlib
import numpy as np
import orbits

from pathlib import Path
from scipy.optimize import fsolve


def trapezoid_method(f, x_0, dt):
	return lambda x_1: x_0 + (f(x_0) + f(x_1)) * dt / 2 - x_1


class BDF2:
	def __init__(self, orbiters: list[orbits.Orbiter], planets: list[orbits.Planet], step_size: float):
		orbiters.sort(key=lambda x: x.id)
		planets.sort(key=lambda x: x.id)

		bodies = orbiters + planets
		self.body_names = [i.name for i in bodies]
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

	def __str__(self):
		return "BDF2"

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


class RK4:
	def __init__(self, orbiters: list[orbits.Orbiter], planets: list[orbits.Planet], step_size: float):
		orbiters.sort(key=lambda x: x.id)
		planets.sort(key=lambda x: x.id)

		bodies = orbiters + planets
		self.body_names = [i.name for i in bodies]
		self.state = [np.zeros(6 * len(bodies))]
		for n, bod in enumerate(bodies):
			self.state[0][n * 6:(n + 1) * 6] = [*bod.position, *bod.velocity]

		self.grav_param_mat = np.array([i.grav_param for i in planets])
		self.n_orbiters = len(orbiters)
		self.step_size = step_size

		self.t = 0

		self._state_array = None

		self.tableau = np.array([[0.5] * 6 * len(bodies), [0.5] * 6 * len(bodies), [1] * 6 * len(bodies)])

	def __str__(self):
		return "RK4"

	def _f(self, state):
		return orbits.f_with_n_orbiters(state, self.grav_param_mat, self.n_orbiters)

	def step(self):
		self.state.append(self.state[-1] + self.step_size * (
				self._f(self.state[-1]) / 6 +
				self._f(self.state[-1] + self.tableau[0] * self.step_size) / 3 +
				self._f(self.state[-1] + self.tableau[1] * self.step_size) / 3 +
				self._f(self.state[-1] + self.step_size) / 6
		))
		self.t += self.step_size

	@property
	def state_array(self):
		if self._state_array is None:
			self._state_array = np.array(self.state)

		return self._state_array


class DummyIntegrator:
	def __init__(self, int_type, state, names, n_orbiters, step_size, t):
		self._int_type = int_type
		self.state_array = state
		self.body_names = list(names)
		self.n_orbiters = n_orbiters
		self.step_size = step_size
		self.t = t

	def __str__(self):
		return self._int_type


class IntegratorData:
	def __init__(self, integrator: RK4 | BDF2 | DummyIntegrator):
		self.integrator = integrator
		i_earth = self.integrator.body_names.index('Earth')
		if i_earth == -1:
			print("Plot could not find Earth in integrator bodies, assuming ECI reference")

		orb_pos_arrays = [self.integrator.state_array[:, i:i + 3] for i in range(0, self.integrator.n_orbiters * 6, 6)]
		earth_pos = self.integrator.state_array[:, i_earth * 6: i_earth * 6 + 3]

		self.orb_pos_eci = [pos - earth_pos for pos in orb_pos_arrays]

	def __str__(self):
		return f"{str(self.integrator)}, dt = {self.integrator.step_size}"

	@classmethod
	def load_data(cls, path: Path):
		data = np.load(path, allow_pickle=False)
		try:
			return cls(
				integrator=DummyIntegrator(str(data["arr_0"]), data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"],
				                           data["arr_5"]))
		except KeyError:
			print("Malformed data")

	def plot_orbiters(self, axes: matplotlib.axes.Axes, step_size: float = 120):
		step_iter = step_size / self.integrator.step_size
		if step_iter % 1 != 0:
			print("Plot step size not multiple of integrator step, rounding down to nearest integer")

		step_iter = int(step_iter)

		plot_orb_pos_eci = [case[::step_iter] for case in self.orb_pos_eci]

		for pos, name in zip(plot_orb_pos_eci, self.integrator.body_names[:self.integrator.n_orbiters]):
			axes.plot(*pos.transpose(), label=name)

		axes.set_title(f"{str(self.integrator)}, dt = {self.integrator.step_size}")
		axes.legend()

	def calc_norm_metric(self):
		l_mag_t0 = [np.linalg.norm(state[0]) for state in self.orb_pos_eci]

		max_mag_err = [np.max(np.abs(np.array([np.linalg.norm(t) for t in all_t]) - t0)) for t0, all_t in zip(l_mag_t0, self.orb_pos_eci)]

		return max_mag_err

	def calc_plane_metric(self):
		max_out_of_plane = [np.max(state[:, 2]) for state in self.orb_pos_eci]

		return max_out_of_plane

	def cache_data(self):
		cache_path = Path(".") / 'cache' / f"{str(self.integrator)}_{str(self.integrator.step_size)}_{str(self.integrator.t)}.npy"
		if cache_path.exists():
			print("Cache file already exists, overwriting")
			cache_path.unlink()

		data_array = [str(self.integrator), np.array(self.integrator.state_array), self.integrator.body_names, self.integrator.n_orbiters, self.integrator.step_size, self.integrator.t]
		np.savez(cache_path, *data_array, allow_pickle=False)

	def is_same(self, integrator: RK4 | BDF2, end_time: float) -> bool:
		return self.integrator.step_size == integrator.step_size and str(self.integrator) == str(integrator) and abs(self.integrator.t - end_time) < (self.integrator.step_size + 1)

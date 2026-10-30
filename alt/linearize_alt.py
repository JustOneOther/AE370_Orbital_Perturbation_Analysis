"""
Handles the linearization of the n-body system and the stability of candidate methods

State order matches orbits.build_state: orbiters first, then planets
"""
from scipy.optimize import brentq

import numpy as np
import sympy as sp


N_ZERO = 6  # Translation zeros, moving every body by the same position or velocity changes nothing


def jacobian(n_orbiters: int, n_planets: int):
	"""
	Jacobian of orbits.f_with_n_orbiters from SymPy, returned as a function of (state, grav_param_mat)
	"""
	n_bodies = n_orbiters + n_planets
	state = sp.symbols(f"x_0:{6 * n_bodies}", real=True)
	grav_param = sp.symbols(f"mu_0:{n_planets}", real=True)

	f = []
	for i in range(n_bodies):
		start = 6 * i
		accel = sp.zeros(3, 1)

		for j in range(n_planets):
			bod_start = 6 * (j + n_orbiters)
			if bod_start == start: continue

			r = sp.Matrix(state[bod_start:bod_start + 3]) - sp.Matrix(state[start:start + 3])

			accel += grav_param[j] * r / r.norm() ** 3

		f += list(state[start + 3:start + 6]) + list(accel)

	jac = sp.Matrix(f).jacobian(sp.Matrix(state))

	return sp.lambdify((state, grav_param), jac, "numpy")


def owner(eigenvector: np.ndarray, n_orbiters: int) -> int:
	"""
	Index of the body whose 6 entries dominate an eigenvector, with planets taking priority
	"""
	weights = np.array([np.linalg.norm(eigenvector[6 * b:6 * b + 6]) for b in range(eigenvector.size // 6)])

	# Orbiters ride along with any mode that moves a planet, so credit the planet
	if weights[n_orbiters:].sum() > 1e-8 * weights.sum():
		return n_orbiters + int(np.argmax(weights[n_orbiters:]))

	return int(np.argmax(weights))


def modes(jac: np.ndarray, n_orbiters: int):
	"""
	Eigenvalues, the body each belongs to, the indices of the real modes, and the indices of the translation zeros
	"""
	eigenvalues, eigenvectors = np.linalg.eig(jac)
	owners = np.array([owner(eigenvectors[:, j], n_orbiters) for j in range(eigenvalues.size)])
	order = np.argsort(abs(eigenvalues))

	return eigenvalues, owners, order[N_ZERO:], order[:N_ZERO]


def stiffness_ratio(eigenvalues: np.ndarray) -> float:
	"""
	Lecture 13 stiffness ratio max|Re| / min|Re|, which is meaningless here since Re = 0 on the imaginary pairs
	"""
	with np.errstate(divide="ignore"):  # min|Re| can be exactly 0
		return max(abs(eigenvalues.real)) / min(abs(eigenvalues.real))


def roots_fe(w):
	"""
	Root of the forward Euler stability polynomial at w = dt * lambda
	"""
	return np.array([1 + w])


def roots_rk4(w):
	"""
	Root of the RK4 stability polynomial at w = dt * lambda
	"""
	return np.array([1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24])


def roots_trap(w):
	"""
	Root of the trapezoid stability polynomial at w = dt * lambda
	"""
	with np.errstate(divide="ignore", invalid="ignore"):  # Pole at w = 2
		return np.array([(1 + w / 2) / (1 - w / 2)])


def roots_bdf2(w):
	"""
	Roots of the BDF2 stability polynomial (3 - 2w) zeta^2 - 4 zeta + 1 at w = dt * lambda, principal root first
	"""
	s = np.sqrt(1 + 2 * w + 0j)
	with np.errstate(divide="ignore", invalid="ignore"):  # Pole at w = 3/2
		return np.array([(2 + s) / (3 - 2 * w), (2 - s) / (3 - 2 * w)])


METHODS = {"forward Euler": roots_fe, "RK4": roots_rk4, "trapezoid": roots_trap, "BDF2": roots_bdf2}


def amplification(roots, w):
	"""
	Largest |zeta| at w, the step is absolutely stable where this is <= 1
	"""
	return np.max(np.abs(roots(w)), axis=0)


def stability_eigs(eigenvalues: np.ndarray) -> np.ndarray:
	"""
	Eigenvalues a stable method must keep bounded, with roundoff real parts zeroed and growing modes dropped
	"""
	eigenvalues = np.where(abs(eigenvalues.real) <= 1e-12 * abs(eigenvalues), 1j * eigenvalues.imag, eigenvalues)

	return eigenvalues[eigenvalues.real <= 0]


def _max_amplification(roots, dt, eigenvalues):
	return np.max(amplification(roots, dt * eigenvalues))


def stable_dt(roots, eigenvalues: np.ndarray, dts=np.geomspace(1e-3, 1e7, 4001)) -> float:
	"""
	Largest stable dt, np.inf if there is no limit and 0.0 if there is no stable dt
	"""
	stable = np.array([_max_amplification(roots, dt, eigenvalues) <= 1 + 1e-14 for dt in dts])
	if stable.all(): return np.inf
	if not stable[0]: return 0.0

	dt_lo, dt_hi = dts[np.argmin(stable) - 1], dts[np.argmin(stable)]

	# Solve only on the eigenvalues that cross, the slow modes give |zeta| = 1.0 to machine precision
	crossing = eigenvalues[amplification(roots, dt_hi * eigenvalues) > 1 + 1e-14]

	return brentq(lambda dt: _max_amplification(roots, dt, crossing) - 1, dt_lo, dt_hi, xtol=1e-10)
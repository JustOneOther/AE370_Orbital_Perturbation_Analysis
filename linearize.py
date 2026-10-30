import numpy as np
import sympy as sp


def jacobian(n):

	x = sp.symbols(f"x_0:{6 * n}", real=True)
	mu = sp.symbols(f"mu_0:{n}", real=True)

	f = []
	for i in range(n):
		r_i = sp.Matrix(x[6*i:6 * i + 3])
		v_i = sp.Matrix(x[6 * i + 3:6 * i + 6])
		a_i = sp.zeros(3, 1)
		for k in range(n):
			if k == i:
				continue
			d = sp.Matrix(x[6 * k:6 * k + 3]) - r_i
			a_i += mu[k] * d / d.norm() ** 3
		f += list(v_i) + list(a_i)

	J = sp.Matrix(f).jacobian(sp.Matrix(x))
	J_num = sp.lambdify((x, mu), J, "numpy")

	return J_num



N_ZERO = 6 

def owner(v: np.ndarray, n_massive: int) -> int:
	"""
	Index of the body a mode (eigenvector v) belongs to: the body whose 6 entries dominate v

	Orbiters cannot move planets, so a mode that moves any planet is credited to the planet.
	Otherwise Earth's modes split evenly between Earth and the orbiters riding along with it.
	"""
	n = v.size // 6
	w = np.array([np.linalg.norm(v[6 * b:6 * b + 6]) for b in range(n)])
	if w[:n_massive].sum() > 1e-8 * w.sum():
		return int(np.argmax(w[:n_massive]))
	return int(np.argmax(w))


def modes(J: np.ndarray, n_massive: int):
	"""
	Eigenvalues of J, which body each belongs to, and which are the translation zeros

	Returns
	-------
	ev : ndarray, eigenvalues
	owners : ndarray of int, body index of each eigenvalue
	mode_idx : ndarray of int, indices of the real modes (translation zeros removed)
	zero_idx : ndarray of int, indices of the N_ZERO translation zeros
	"""
	ev, vec = np.linalg.eig(J)
	owners = np.array([owner(vec[:, j], n_massive) for j in range(ev.size)])
	order = np.argsort(abs(ev))
	return ev, owners, order[N_ZERO:], order[:N_ZERO]


def stiffness_ratio(ev: np.ndarray) -> float:
	"""
	Lecture 13 ratio max|Re lambda| / min|Re lambda|, as in Project 1. Meaningless here: the
	imaginary pairs have Re lambda = 0 to roundoff, so it comes out huge or infinite
	"""
	with np.errstate(divide="ignore"):
		return max(abs(ev.real)) / min(abs(ev.real))


# ---------------------------------------------------------------- stability limits (the methods themselves are in methods.py)

def stability_eigs(lam: np.ndarray) -> np.ndarray:
	"""
	Eigenvalues a stable method must keep bounded

	Real parts below 1e-12 |lambda| are roundoff on the imaginary pairs (~1e-20) and are set to 0;
	left in, they make the trapezoid method look unstable at huge dt. Growing modes (Re > 0) are
	dropped: the linearized solution really grows there, so that is an accuracy question.
	"""
	lam = np.where(abs(lam.real) <= 1e-12 * abs(lam), 1j * lam.imag, lam)
	return lam[lam.real <= 0]


def stable_dt(amp, lam: np.ndarray, dts=np.geomspace(1e-3, 1e7, 4001)) -> float:
	"""
	Largest dt in dts with every w = dt * lambda inside the method's stability region (|zeta| <= 1)

	amp : a method's amplification function from methods.py, e.g. methods.amp_rk4

	Returns np.inf if stable at every dt tested, 0.0 if unstable at the smallest
	"""
	stable = np.array([np.all(amp(dt * lam) <= 1 + 1e-14) for dt in dts])
	if stable.all():
		return np.inf
	if not stable[0]:
		return 0.0
	return float(dts[np.argmin(stable) - 1])
"""
Time-marching methods and their stability polynomials

Each marcher uses the Project 1 convention:  u, times = method(f, u_start, t_end, dt)
Each amp_* function applies the method to u' = lambda u and returns the largest |zeta| at w = dt * lambda.
The step is absolutely stable where that is <= 1.
"""
import numpy as np
from scipy.optimize import fsolve


# ---------------------------------------------------------------- forward Euler

def forward_euler(f, u_start, t_end, dt):
	'''March u' = f(u, t) from t = 0 to t_end with forward Euler. Returns (u, times).'''
	n = int(round(t_end / dt))
	u = np.empty((n + 1, np.size(u_start)))
	u[0] = u_start
	for k in range(n):
		u[k + 1] = u[k] + dt * f(u[k], k * dt)
	return u, np.arange(n + 1) * dt


def amp_fe(w):
	'''zeta = 1 + w'''
	return abs(1 + w)


# ---------------------------------------------------------------- RK4

def rk4(f, u_start, t_end, dt):
	'''March u' = f(u, t) from t = 0 to t_end with classical RK4. Returns (u, times).'''
	n = int(round(t_end / dt))
	u = np.empty((n + 1, np.size(u_start)))
	u[0] = u_start
	for k in range(n):
		t = k * dt
		k1 = f(u[k], t)
		k2 = f(u[k] + 0.5 * dt * k1, t + 0.5 * dt)
		k3 = f(u[k] + 0.5 * dt * k2, t + 0.5 * dt)
		k4 = f(u[k] + dt * k3, t + dt)
		u[k + 1] = u[k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
	return u, np.arange(n + 1) * dt


def amp_rk4(w):
	'''zeta = 1 + w + w^2/2 + w^3/6 + w^4/24'''
	return abs(1 + w + w**2 / 2 + w**3 / 6 + w**4 / 24)


# ---------------------------------------------------------------- trapezoid

def trapezoid(f, u_start, t_end, dt, jac=None):
	'''March u' = f(u, t) with the trapezoid method, solving each step with fsolve.

	jac : optional function v -> df/du at v. Passed to fsolve as the exact Newton derivative
	(I - dt/2 J) instead of a finite-difference estimate. Returns (u, times).
	'''
	n = int(round(t_end / dt))
	u = np.empty((n + 1, np.size(u_start)))
	u[0] = u_start
	I = np.eye(np.size(u_start))
	fp = None if jac is None else (lambda v: I - dt / 2 * jac(v))
	for k in range(n):
		fk = f(u[k], k * dt)
		g = lambda v: v - u[k] - dt / 2 * (fk + f(v, (k + 1) * dt))
		u[k + 1] = fsolve(g, u[k] + dt * fk, fprime=fp)
	return u, np.arange(n + 1) * dt


def amp_trap(w):
	'''zeta = (1 + w/2) / (1 - w/2)'''
	with np.errstate(divide="ignore", invalid="ignore"):  # pole at w = 2
		return abs((1 + w / 2) / (1 - w / 2))


# ---------------------------------------------------------------- BDF2

def bdf2(f, u_start, t_end, dt, jac=None):
	'''March u' = f(u, t) with BDF2, started with one trapezoid step, solving each step with fsolve.

	jac : optional function v -> df/du at v, used as the Newton derivative (I - 2/3 dt J).
	Returns (u, times).
	'''
	n = int(round(t_end / dt))
	u = np.empty((n + 1, np.size(u_start)))
	u[0] = u_start
	I = np.eye(np.size(u_start))
	u[1] = trapezoid(f, u_start, dt, dt, jac)[0][1]  # one trapezoid starter step
	fp = None if jac is None else (lambda v: I - 2 / 3 * dt * jac(v))
	for k in range(1, n):
		rhs = (4 * u[k] - u[k - 1]) / 3
		g = lambda v: v - rhs - 2 / 3 * dt * f(v, (k + 1) * dt)
		u[k + 1] = fsolve(g, u[k], fprime=fp)
	return u, np.arange(n + 1) * dt


def amp_bdf2(w):
	'''Largest root of (3 - 2w) zeta^2 - 4 zeta + 1 = 0, i.e. (2 +/- sqrt(1 + 2w)) / (3 - 2w)'''
	s = np.sqrt(1 + 2 * w + 0j)
	with np.errstate(divide="ignore", invalid="ignore"):  # pole at w = 3/2
		return np.maximum(abs((2 + s) / (3 - 2 * w)), abs((2 - s) / (3 - 2 * w)))


# ---------------------------------------------------------------- all methods in one place

METHODS = {
	"forward Euler": {"march": forward_euler, "amp": amp_fe},
	"RK4":           {"march": rk4,           "amp": amp_rk4},
	"trapezoid":     {"march": trapezoid,     "amp": amp_trap},
	"BDF2":          {"march": bdf2,          "amp": amp_bdf2},
}

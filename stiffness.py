"""This file contains the linearization and stiffness analysis, run separately from main.py before choosing the method"""
import dynamics
import linearize
import matplotlib.pyplot as plt
import numpy as np
import orbits


# Constants
TIMESTEP = 10  # Step marked on the stability regions
N_PER_ORBIT = [10, 50, 100]  # Steps per LEO orbit for the accuracy table
UNITS = [("yr", 3.15576E+07), ("days", 86400), ("h", 3600), ("min", 60)]

# Roots of each method's stability polynomial at w = dt * lambda, principal root first
ROOTS = {
	"Forward Euler": lambda w: [1 + w],
	"RK4": lambda w: [1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24],
	"Trapezoid": lambda w: [(1 + w / 2) / (1 - w / 2)],
	"BDF2": lambda w: [(2 + np.sqrt(1 + 2 * w + 0j)) / (3 - 2 * w), (2 - np.sqrt(1 + 2 * w + 0j)) / (3 - 2 * w)],
}


def amplification(name: str):
	def amp(w):
		with np.errstate(divide="ignore", invalid="ignore"):  # Poles at w = 2 (trapezoid) and w = 3/2 (BDF2)
			return np.max(np.abs(ROOTS[name](w)), axis=0)

	return amp


def readable(seconds: float) -> str:
	"""Seconds in the largest unit that keeps the number at or above 1"""
	for unit, size in UNITS:
		if seconds >= size: return f"{seconds / size:.3g} {unit}"

	return f"{seconds:.3g} s"


def refined_dt(amp, lam: np.ndarray) -> float:
	"""Largest stable dt from linearize.stable_dt, refined on a fine grid around the first estimate"""
	dt_max = linearize.stable_dt(amp, lam)
	if dt_max in (0.0, np.inf): return dt_max

	return linearize.stable_dt(amp, lam, np.linspace(dt_max, dt_max * 1.01, 10001))


# Setup bodies, planets first then orbiters (dynamics.build)

planets, orbiters = orbits.get_starting_bodies()
state_0, grav_param, names = dynamics.build(planets, orbiters)
n_massive = len(planets)


# Jacobian at the initial state, and its eigenvalues by body

jacobian = linearize.jacobian(len(names))
J_0 = jacobian(state_0, grav_param)
ev, owners, mode_idx, zero_idx = linearize.modes(J_0, n_massive)

by_body = [[j for j in mode_idx if owners[j] == b] for b in range(len(names))]

print(f"{'body':12s} {'max Re(lam)':>12} {'max Im(lam)':>12}   (1/s)")
for name, mine in zip(names, by_body):
	if not mine:
		print(f"{name:12s} {'0':>12} {'0':>12}")  # Its 6 eigenvalues are the translation zeros
		continue

	print(f"{name:12s} {max(abs(ev[mine].real)):12.3e} {max(abs(ev[mine].imag)):12.3e}")
print(f"translation zeros: |lam| <= {abs(ev[zero_idx]).max():.1e}\n")

# # prints for table paper

# rows = []
# for name, mine in zip(names, by_body):
# 	label = name.replace(" Orbiter", "")
# 	if not mine:
# 		rows.append((0.0, label, "0", "0"))
# 		continue

# 	re = max(abs(ev[mine].real))  # The real pair, +/- sqrt(2) n
# 	im = max(abs(ev[mine].imag))  # The two imaginary pairs, +/- i n
# 	rows.append((max(re, im), label, f"$\\pm${re:.2E}", f"$\\pm${im:.2E}$\\,i$"))

# rows.sort(key=lambda r: r[0], reverse=True)

# print(" & ".join(r[1] for r in rows) + r" \\")
# print(r"\midrule")
# print(" & ".join(r[2] for r in rows) + r" \\")
# print(" & ".join(r[3] for r in rows) + r" \\" + "\n")


# Timescales of each body's modes, 1 / |lambda| (Lecture 12)

print(f"{'body':12s} {'real pair 1/|lam|':>18} {'imag pairs 1/|lam|':>19} {'period 2pi/|Im lam|':>20}")
for name, mine in zip(names, by_body):
	if not mine:
		print(f"{name:12s} {'none':>18} {'none':>19} {'none':>20}")
		continue

	tau_real = 1 / max(abs(ev[mine].real))  # The real pair, sqrt(2) n
	tau_imag = 1 / max(abs(ev[mine].imag))  # The imaginary pairs, n
	print(f"{name:12s} {readable(tau_real):>18} {readable(tau_imag):>19} {readable(2 * np.pi * tau_imag):>20}")
print()


# Stiffness ratio (Lecture 13), max|lambda| / min|lambda| with the translation zeros excluded
# linearize.stiffness_ratio uses Re(lambda), which divides by the roundoff on the imaginary pairs

tau = 1 / abs(ev[mode_idx])
fast, slow = mode_idx[np.argmin(tau)], mode_idx[np.argmax(tau)]
p_leo = 2 * np.pi / max(abs(ev[mode_idx].imag))  # Period of the fastest orbit

print(f"fastest mode  |lam| = {abs(ev[fast]):.3e} 1/s   timescale {readable(1 / abs(ev[fast])):>9}   ({names[owners[fast]]})")
print(f"slowest mode  |lam| = {abs(ev[slow]):.3e} 1/s   timescale {readable(1 / abs(ev[slow])):>9}   ({names[owners[slow]]})")
print(f"R_s = max|lam| / min|lam| = {max(abs(ev[mode_idx])) / min(abs(ev[mode_idx])):.3e}\n")


# Largest stable step of each candidate method

lam = linearize.stability_eigs(ev[mode_idx])

print(f"{'method':14s} {'largest stable dt':>18} {'steps per LEO orbit':>20}")
for name in ROOTS:
	dt_max = refined_dt(amplification(name), lam)
	if dt_max == np.inf: print(f"{name:14s} {'no limit':>18}")
	elif dt_max == 0.0: print(f"{name:14s} {'none':>18}")
	else: print(f"{name:14s} {dt_max:16.2f} s {p_leo / dt_max:20.1f}")
print()

# # latex print for table in paper

# POLYNOMIALS = {
# 	"BDF2": r"$(3 - 2w)\,\zeta^2 - 4\zeta + 1$",
# 	"RK4": r"$\zeta - \left(1 + w + \tfrac{w^2}{2} + \tfrac{w^3}{6} + \tfrac{w^4}{24}\right)$",
# 	"Forward Euler": r"$\zeta - (1 + w)$",
# 	"Trapezoid": r"$\left(1 - \tfrac{w}{2}\right)\zeta - \left(1 + \tfrac{w}{2}\right)$",
# }

# for name, poly in POLYNOMIALS.items():
# 	dt_max = refined_dt(amplification(name), lam)
# 	if dt_max == np.inf: dt_text, steps = "No limit", ""
# 	elif dt_max == 0.0: dt_text, steps = "None", ""
# 	else: dt_text, steps = f"{dt_max:.0f}~s", f"{p_leo / dt_max:.1f}"

# 	print(f"{name[0].upper() + name[1:]} & {poly} & {dt_text} & {steps} \\\\")
# print()


# Accuracy on the orbital oscillation: model problem w = i 2 pi / N, error per orbit

dt_rk4 = refined_dt(amplification("RK4"), lam)

print(f"{'N':>6} " + " ".join(f"{m + ' phase':>16} {m + ' amp':>14}" for m in ["RK4", "trapezoid", "BDF2"]))
for N in [p_leo / dt_rk4] + N_PER_ORBIT:
	w = 1j * 2 * np.pi / N
	line = f"{N:6.1f} "

	for name in ["RK4", "Trapezoid", "BDF2"]:
		zeta = ROOTS[name](w)[0]  # Principal root
		line += f"{N * np.angle(zeta) - 2 * np.pi:16.1e} {abs(zeta) ** N - 1:14.1e} "
	print(line)


# Stability regions, one figure per method, with w = dt * lambda for every mode a stable method must keep bounded
# The positive real eigenvalues are left off: the linearized solution grows there, so no method should keep them bounded

x, y = np.meshgrid(np.linspace(-4, 3.5, 401), np.linspace(-3.5, 3.5, 401))
dt_lam = x + 1j * y

for name in ROOTS:
	stable = amplification(name)(dt_lam) <= 1

	fig, ax = plt.subplots(figsize=(4.6, 3.0))  # Sized for a half-width subfigure in the paper
	ax.contourf(x, y, stable, levels=[0.5, 1.5], colors='orange')

	w = TIMESTEP * lam
	ax.plot(w.real, w.imag, 'kx', label=fr'$\Delta t$ = {TIMESTEP:g} s')

	w = dt_rk4 * lam
	ax.plot(w.real, w.imag, 'ko', mfc='none', label=fr'$\Delta t$ = {dt_rk4:.0f} s')

	ax.axhline(0, color='k', lw=0.5)
	ax.axvline(0, color='k', lw=0.5)
	ax.set_title(name)
	ax.set_xlabel(r'Re($\Delta t\,\lambda$)')
	ax.set_ylabel(r'Im($\Delta t\,\lambda$)')
	ax.legend(fontsize=8, loc='upper left')
	plt.tight_layout()

plt.show()

"""This file contains the linearization and stiffness analysis, run before choosing the method"""
import linearize_alt as linearize
import matplotlib.pyplot as plt
import numpy as np
import orbits_alt as orbits


# Constants
TIMESTEP = 10  # Step used in main.py, marked on the stability regions
N_PER_ORBIT = [10, 50, 100]  # Steps per LEO orbit for the accuracy table


planets, orbiters = orbits.get_starting_bodies()
state_0, grav_param_mat = orbits.build_state(orbiters, planets)
names = [o.name for o in sorted(orbiters, key=lambda x: x.id)] + [p.name for p in sorted(planets, key=lambda x: x.id)]
n_orbiters = len(orbiters)


# Jacobian at the initial state, and its eigenvalues by body
jacobian = linearize.jacobian(n_orbiters, len(planets))
J_0 = jacobian(state_0, grav_param_mat)
ev, owners, mode_idx, zero_idx = linearize.modes(J_0, n_orbiters)

print(f"{'body':12s} {'max Re(lam)':>12} {'max Im(lam)':>12}   (1/s)")
for b, name in enumerate(names):
	mine = [j for j in mode_idx if owners[j] == b]
	if not mine:
		print(f"{name:12s} {'none':>12} {'none':>12}")
		continue
	print(f"{name:12s} {max(abs(ev[mine].real)):12.3e} {max(abs(ev[mine].imag)):12.3e}")
print(f"translation zeros: |lam| <= {abs(ev[zero_idx]).max():.1e}\n")


# LaTeX rows for the eigenvalue table, descending |lambda|
rows = []
for b, name in enumerate(names):
	mine = [j for j in mode_idx if owners[j] == b]
	label = name.replace(" Orbiter", "")
	if not mine:
		rows.append((0.0, label, "none", "none"))
		continue
	re = max(abs(ev[mine].real))  # The real pair, +/- sqrt(2) n
	im = max(abs(ev[mine].imag))  # The two imaginary pairs, +/- i n
	rows.append((max(re, im), label, f"$\\pm${re:.2E}", f"$\\pm${im:.2E}$\\,i$"))

rows.sort(key=lambda r: r[0], reverse=True)

print(" & ".join(r[1] for r in rows) + r" \\")
print(r"\midrule")
print(" & ".join(r[2] for r in rows) + r" \\")
print(" & ".join(r[3] for r in rows) + r" \\" + "\n")


# Stiffness ratio and timescale spread
mag = abs(ev[mode_idx])
fast, slow = mode_idx[np.argmax(mag)], mode_idx[np.argmin(mag)]
P_LEO = 2 * np.pi / max(abs(ev[mode_idx].imag))  # Period of the fastest orbit

print(f"R_s (Lecture 13)  = {linearize.stiffness_ratio(ev):.1e}")
print(f"fastest |lam|     = {abs(ev[fast]):.3e} 1/s   ({names[owners[fast]]}, timescale {2 * np.pi / abs(ev[fast]) / 60:.1f} min)")
print(f"slowest |lam|     = {abs(ev[slow]):.3e} 1/s   ({names[owners[slow]]}, timescale {2 * np.pi / abs(ev[slow]) / 3.15576e7:.1f} yr)")
print(f"timescale spread  = {abs(ev[fast]) / abs(ev[slow]):.2e}\n")


# Largest stable step of each candidate method
lam = linearize.stability_eigs(ev[mode_idx])

print(f"{'method':14s} {'largest stable dt':>18} {'steps per LEO orbit':>20}")
for name, roots in linearize.METHODS.items():
	dt_max = linearize.stable_dt(roots, lam)
	if dt_max == np.inf: print(f"{name:14s} {'no limit':>18}")
	elif dt_max == 0.0: print(f"{name:14s} {'none':>18}")
	else: print(f"{name:14s} {dt_max:16.2f} s {P_LEO / dt_max:20.1f}")
print()


# Accuracy on the orbital oscillation: model problem w = i 2 pi / N, error per orbit
dt_rk4 = linearize.stable_dt(linearize.roots_rk4, lam)

print(f"{'N':>6} " + " ".join(f"{m + ' phase':>16} {m + ' amp':>14}" for m in ["RK4", "trapezoid", "BDF2"]))
for N in [P_LEO / dt_rk4] + N_PER_ORBIT:
	w = 1j * 2 * np.pi / N
	line = f"{N:6.1f} "
	for roots in [linearize.roots_rk4, linearize.roots_trap, linearize.roots_bdf2]:
		zeta = roots(w)[0]  # Principal root
		line += f"{N * np.angle(zeta) - 2 * np.pi:16.1e} {abs(zeta) ** N - 1:14.1e} "
	print(line)


# Stability regions, one figure per method, with w = dt * lambda for every mode
w_acc = TIMESTEP * ev[mode_idx]
w_lim = dt_rk4 * ev[mode_idx]

x, y = np.meshgrid(np.linspace(-4, 3.5, 401), np.linspace(-3.5, 3.5, 401))
dt_lam = x + 1j * y

for name, roots in linearize.METHODS.items():
	stable = linearize.amplification(roots, dt_lam) <= 1

	fig, ax = plt.subplots(figsize=(9, 4))
	ax.contourf(x, y, stable, levels=[0.5, 1.5], colors='orange')
	ax.plot(w_acc.real, w_acc.imag, 'kx', label=fr'$\Delta t$ = {TIMESTEP:.0f} s')
	ax.plot(w_lim.real, w_lim.imag, 'ko', mfc='none', label=fr'$\Delta t$ = {dt_rk4:.0f} s')
	ax.axhline(0, color='k', lw=0.5)
	ax.axvline(0, color='k', lw=0.5)
	ax.set_title(name)
	ax.set_xlabel(r'Re($\Delta t\,\lambda$)')
	ax.set_ylabel(r'Im($\Delta t\,\lambda$)')
	ax.legend()
	plt.tight_layout()

plt.show()

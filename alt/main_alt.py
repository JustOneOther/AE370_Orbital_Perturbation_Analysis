"""This file contains the code that is to run the analysis"""
import integrators_alt as integrators
import matplotlib.pyplot as plt
import numpy as np
import orbits_alt as orbits
import time


# Constants
N_DAYS = 30
TOTAL_TIME = N_DAYS * 24 * 60 * 60  # Step through 30 days
TIMESTEP = 10  # Ten second time step, set by the LEO numerical error
N_STEPS = TOTAL_TIME // TIMESTEP
UPDATE_INTERVAL = 0.1
UPDATE_STEP = int(N_STEPS * UPDATE_INTERVAL)
TOLERANCE = 1  # km, how far the Kepler prediction may be off


planets, orbiters = orbits.get_starting_bodies()
earth = [p for p in planets if p.name == "Earth"]

# Full n-body system, and Earth alone: the Earth-only run has an exact solution, the Kepler orbit,
# so its difference from Kepler is the numerical error of the method at this step size
full = integrators.RK4(orbiters, planets, TIMESTEP)
earth_only = integrators.RK4(orbiters, earth, TIMESTEP)

print(f"Stepping for {N_STEPS} steps\n")

for integrator, label in [(full, "n-body"), (earth_only, "Earth only")]:
	start_time = time.time()

	for i in range(N_STEPS):
		if i % UPDATE_STEP == 0 and i != 0:
			print(f"{label}: running step {i}, estimated time to completion: {(N_STEPS - i) * (time.time() - start_time) / i / 60:.1f} mins")

		integrator.step()


times = np.arange(N_STEPS + 1) * TIMESTEP
days = times / (24 * 60 * 60)

orbiters.sort(key=lambda x: x.id)
planets.sort(key=lambda x: x.id)
n_orb = len(orbiters)
earth_full = 6 * (n_orb + planets.index(earth[0]))  # Start of Earth's entries in each state
earth_alone = 6 * n_orb
mu_earth = earth[0].grav_param

f = plt.figure(figsize=(9, 5))
ax = f.add_subplot(1, 1, 1)

for k, (orb, color) in enumerate(zip(orbiters, ['black', 'blue', 'red'])):
	# Positions relative to Earth
	rel_full = full.state_array[:, 6 * k:6 * k + 3] - full.state_array[:, earth_full:earth_full + 3]
	rel_alone = earth_only.state_array[:, 6 * k:6 * k + 3] - earth_only.state_array[:, earth_alone:earth_alone + 3]

	r_0 = rel_full[0]
	v_0 = full.state_array[0, 6 * k + 3:6 * k + 6] - full.state_array[0, earth_full + 3:earth_full + 6]
	kepler = orbits.kepler_position(r_0, v_0, mu_earth, times)

	difference = np.linalg.norm(rel_full - kepler, axis=1)
	numerical_error = np.linalg.norm(rel_alone - kepler, axis=1)

	exceeds = days[np.argmax(difference > TOLERANCE)] if difference.max() > TOLERANCE else np.nan
	print(f"\n{orb.name}: n-body minus Kepler after {N_DAYS} days = {difference[-1]:.3f} km, "
		f"numerical error = {numerical_error.max():.2e} km, exceeds {TOLERANCE} km at {exceeds:.2f} days")

	ax.plot(days[1:], difference[1:], color=color, linestyle='-', linewidth=1, label=orb.name)
	ax.plot(days[1:], numerical_error[1:], color=color, linestyle=':', linewidth=0.75, label=f"{orb.name}, numerical error")

ax.axhline(TOLERANCE, color='k', linestyle='--', linewidth=0.75, label=f"{TOLERANCE} km")
ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel(r'$t$ [days]', fontsize=14)
ax.set_ylabel(r'$|\vec r_{n\mathrm{-body}} - \vec r_{\mathrm{Kepler}}|$ [km]', fontsize=14)
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

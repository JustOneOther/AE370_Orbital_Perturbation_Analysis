"""This file contains the code that is to run the analysis"""
import sys
from pathlib import Path

import integrators
import matplotlib.pyplot as plt
import orbits
import time


# Constants
N_DAYS = 10
TOTAL_TIME = N_DAYS * 24 * 60 * 60  # Step through one year

TIMESTEP_LIST = [7.5, 15, 30, 1, 2, 4]
N_STEPS = [int(TOTAL_TIME // STEP) for STEP in TIMESTEP_LIST]
UPDATE_INTERVAL = [0.05, 0.1, 0.2]
UPDATE_STEP = [int(n_step * interval) for n_step, interval in zip(N_STEPS, UPDATE_INTERVAL * 2)]


# Setup integrators

planets, orbiters = orbits.get_starting_bodies()

bdf2_fine = integrators.BDF2(orbiters, planets, TIMESTEP_LIST[0])
bdf2 = integrators.BDF2(orbiters, planets, TIMESTEP_LIST[1])
bdf2_coarse = integrators.BDF2(orbiters, planets, TIMESTEP_LIST[2])

rk4_fine = integrators.RK4(orbiters, planets, TIMESTEP_LIST[3])
rk4 = integrators.RK4(orbiters, planets, TIMESTEP_LIST[4])
rk4_coarse = integrators.RK4(orbiters, planets, TIMESTEP_LIST[5])

integrator_list = [bdf2_fine, bdf2, bdf2_coarse, rk4_fine, rk4, rk4_coarse]


# Load cache

if "--no-cache" in sys.argv:
	cache_data = []
else:
	cache_path = Path(".") / "cache"
	archives = [Path(".") / "cache" / f.name for f in cache_path.iterdir() if f.name.endswith(".npz")]
	cache_data = [integrators.IntegratorData.load_data(i) for i in archives]


# Run integrators

integrator_data: list[integrators.IntegratorData] = []

for i in range(6):
	found_in_cache = False

	for datum in cache_data:
		if datum.is_same(integrator_list[i], TOTAL_TIME):
			print("Found match in cache\n")
			found_in_cache = True
			integrator_data.append(datum)
			break

	if found_in_cache: continue

	start_time = time.time()
	print(f"Stepping {integrator_list[i]} for {N_STEPS[i]} steps\n")

	for j in range(N_STEPS[i]):
		if j % UPDATE_STEP[i] == 0 and j != 0:
			print(f"Running step {j}, estimated time to completion: {(N_STEPS[i] - j) * (time.time() - start_time) / j / 60} mins")

		integrator_list[i].step()

	integrator_data.append(integrators.IntegratorData(integrator_list[i]))
	integrator_data[-1].cache_data()
	print("\n")


# Plot figures

f = plt.figure(figsize=(4000, 6000, 'px'))

axes = [f.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]

[data.plot_orbiters(ax) for data, ax in zip(integrator_data, axes)]


# Print error metrics

for datum in integrator_data:
	norm_err = datum.calc_norm_metric()
	oop_err = datum.calc_plane_metric()
	print(f'Error for {str(datum)}: \n')
	for name, case_norm, case_oop in zip(datum.integrator.body_names[:datum.integrator.n_orbiters], norm_err, oop_err):
		print(f'Case {name}:\n	Maximum error in position magnitude: {case_norm:f} km\n' +
			  f'	Maximum distance out of plane: {case_oop:f} km')
	print("\n")


# Show plots

plt.show()

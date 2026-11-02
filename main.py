"""This file contains the code that is to run the analysis"""
import integrators
import matplotlib.pyplot as plt
import orbits
import time


# Constants
N_DAYS = 10
TOTAL_TIME = N_DAYS * 24 * 60 * 60  # Step through one year

TIMESTEP_LIST = [7.5, 15, 60, 1/8, 1, 8]
N_STEPS = [int(TOTAL_TIME // STEP) for STEP in TIMESTEP_LIST]
UPDATE_INTERVAL = [0.05, 0.1, 0.2]
UPDATE_STEP = [int(n_step * interval) for n_step, interval in zip(N_STEPS, UPDATE_INTERVAL * 2)]


planets, orbiters = orbits.get_starting_bodies()

bdf2_fine = integrators.BDF2(orbiters, planets, TIMESTEP_LIST[0])
bdf2 = integrators.BDF2(orbiters, planets, TIMESTEP_LIST[1])
bdf2_coarse = integrators.BDF2(orbiters, planets, TIMESTEP_LIST[2])

rk4_fine = integrators.RK4(orbiters, planets, TIMESTEP_LIST[3])
rk4 = integrators.RK4(orbiters, planets, TIMESTEP_LIST[4])
rk4_coarse = integrators.RK4(orbiters, planets, TIMESTEP_LIST[5])

integrator_list = [bdf2_fine, bdf2, bdf2_coarse, rk4_fine, rk4, rk4_coarse]


for i in range(6):
	start_time = time.time()
	print(f"Stepping {integrator_list[i]} for {N_STEPS[i]} steps\n")

	for j in range(N_STEPS[i]):
		if j % UPDATE_STEP[i] == 0 and j != 0:
			print(f"Running step {j}, estimated time to completion: {(N_STEPS[i] - j) * (time.time() - start_time) / j / 60} mins")

		integrator_list[i].step()

	print("\n\n")

# Process data

integrator_data = [integrators.IntegratorData(i) for i in integrator_list]

# Plot figures

f = plt.figure(figsize=(4000, 6000, 'px'))

axes = [f.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]

[data.plot_orbiters(ax) for data, ax in zip(integrator_data, axes)]

plt.show()

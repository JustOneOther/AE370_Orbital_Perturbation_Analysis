"""This file contains the code that is to run the analysis"""
import integrators
import matplotlib.pyplot as plt
import numpy as np
import orbits
import time


# Constants
N_DAYS = 1
TOTAL_TIME = N_DAYS * 24 * 60 * 60  # Step through one year

TIMESTEP = 30
N_STEPS = TOTAL_TIME // TIMESTEP
UPDATE_INTERVAL = 0.05
UPDATE_STEP = int(N_STEPS * UPDATE_INTERVAL)

TIMESTEP_COARSE = 2 * TIMESTEP
N_STEPS_COARSE = TOTAL_TIME // TIMESTEP_COARSE
UPDATE_INTERVAL_COARSE = 0.1
UPDATE_STEP_COARSE = int(N_STEPS_COARSE * UPDATE_INTERVAL_COARSE)

TIMESTEP_FINE = TIMESTEP / 32
N_STEPS_FINE = int(TOTAL_TIME // TIMESTEP_FINE)
UPDATE_INTERVAL_FINE = 0.05
UPDATE_STEP_FINE = int(N_STEPS_FINE * UPDATE_INTERVAL_FINE)


planets, orbiters = orbits.get_starting_bodies()

integrator_fine = integrators.BDF2(orbiters, planets, TIMESTEP_FINE)
integrator = integrators.BDF2(orbiters, planets, TIMESTEP)
integrator_coarse = integrators.BDF2(orbiters, planets, TIMESTEP_COARSE)


# start_time = time.time()
# print(f"Stepping for {N_STEPS_FINE} steps\n")
#
# for i in range(N_STEPS_FINE):
# 	if i % UPDATE_STEP_FINE == 0 and i != 0:
# 		print(f"Running step {i}, estimated time to completion: {(N_STEPS_FINE - i) * (time.time() - start_time) / i / 60} mins")
#
# 	integrator_fine.step()
#
# print("\n\n")

start_time = time.time()
print(f"Stepping for {N_STEPS} steps\n")

for i in range(N_STEPS):
	if i % UPDATE_STEP == 0 and i != 0:
		print(f"Running step {i}, estimated time to completion: {(N_STEPS - i) * (time.time() - start_time) / i / 60} mins")

	integrator.step()

print("\n\n")

start_time = time.time()
print(f"Stepping for {N_STEPS_COARSE} steps\n")

for i in range(N_STEPS_COARSE // 2):
	if i % UPDATE_STEP_COARSE == 0 and i != 0:
		print(f"Running step {i}, estimated time to completion: {(N_STEPS_COARSE - i) * (time.time() - start_time) / i / 60} mins")

	integrator_coarse.step()

print("\n\n")


# Process data

orb_1_pos = integrator.state_array[:, 0:3]
orb_2_pos = integrator.state_array[:, 6:9]
orb_3_pos = integrator.state_array[:, 12:15]
sun_pos = integrator.state_array[:, 18:21]
venus_pos = integrator.state_array[:, 24:27]
earth_pos = integrator.state_array[:, 30:33]
mars_pos = integrator.state_array[:, 36:39]
jupiter_pos = integrator.state_array[:, 42:45]
saturn_pos = integrator.state_array[:, 48:51]

coarse_orb_1_pos = integrator_coarse.state_array[:, 0:3]
coarse_orb_2_pos = integrator_coarse.state_array[:, 6:9]
coarse_orb_3_pos = integrator_coarse.state_array[:, 12:15]
coarse_sun_pos = integrator_coarse.state_array[:, 18:21]
coarse_venus_pos = integrator_coarse.state_array[:, 24:27]
coarse_earth_pos = integrator_coarse.state_array[:, 30:33]
coarse_mars_pos = integrator_coarse.state_array[:, 36:39]
coarse_jupiter_pos = integrator_coarse.state_array[:, 42:45]
coarse_saturn_pos = integrator_coarse.state_array[:, 48:51]

pos_arrays = [orb_1_pos, orb_2_pos, orb_3_pos, sun_pos, venus_pos, earth_pos, mars_pos, jupiter_pos, saturn_pos]
coarse_pos_arrays = [coarse_orb_1_pos, coarse_orb_2_pos, coarse_orb_3_pos, coarse_sun_pos, coarse_venus_pos, coarse_earth_pos,
					 coarse_mars_pos, coarse_jupiter_pos, coarse_saturn_pos]

final_pos_arrays = np.array([state[-1] for state in pos_arrays])
final_pos_arrays_coarse = np.array([state[-1] for state in coarse_pos_arrays])

err_arrays = final_pos_arrays - final_pos_arrays_coarse
name_array = ["LEO", "MEO", "GSO", "Sun", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]

max_err = np.max(err_arrays)

pos_arrays = [(pos.transpose(), name) for pos, name in zip(pos_arrays, name_array)]

orb_1_pos_eci = orb_1_pos - earth_pos
orb_2_pos_eci = orb_2_pos - earth_pos
orb_3_pos_eci = orb_3_pos - earth_pos

eci_arrays = [orb_1_pos_eci, orb_2_pos_eci, orb_3_pos_eci]

ref_magnitudes = [np.linalg.norm(state_array[0]) for state_array in eci_arrays]
highest_deviation = [str(np.max(np.array([np.linalg.norm(state) for state in state_array]) - ref_mag))
					 for state_array, ref_mag in zip(eci_arrays, ref_magnitudes)]

print(f"Maximum radius deviation: \nLEO: {highest_deviation[0]}\nMEO: {highest_deviation[1]}\nGSO: {highest_deviation[2]}")

eci_arrays = [(eci.transpose(), name) for eci, name in zip(eci_arrays, name_array[:3])]


# Plot figures

f = plt.figure(figsize=(4000, 2000, 'px'))

ax1 = f.add_subplot(2, 1, 1, projection='3d')
ax2 = f.add_subplot(2, 1, 2, projection='3d')


for pos in pos_arrays: ax1.plot(*(pos[0]), '.-', label=pos[1])

for pos in eci_arrays: ax2.plot(*(pos[0]), '.-', label=pos[1])

ax1.legend()
ax2.legend()

plt.show()

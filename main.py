"""This file contains the code that is to run the analysis"""
import integrators
import matplotlib.pyplot as plt
import orbits
import time


# Constants
N_DAYS = 365
TOTAL_TIME = N_DAYS * 24 * 60 * 60  # Step through one year
TIMESTEP = 60  # One minute time step
N_STEPS = TOTAL_TIME // TIMESTEP
UPDATE_INTERVAL = 0.01
UPDATE_STEP = int(N_STEPS * UPDATE_INTERVAL)

print(f"Stepping for {N_STEPS} steps\n")


planets, orbiters = orbits.get_starting_bodies()

integrator = integrators.BDF2(orbiters, planets, 60)

start_time = time.time()

for i in range(N_STEPS):
	if i % UPDATE_STEP == 0 and i != 0:
		print(f"Running step {i}, estimated time to completion: {(N_STEPS - i) * (time.time() - start_time) / i / 60} mins")

	integrator.step()


f = plt.figure(figsize=(4000, 2000, 'px'))

ax1 = f.add_subplot(2, 1, 1, projection='3d')
ax2 = f.add_subplot(2, 1, 2, projection='3d')

orb_1_pos = integrator.state_array[:, 0:3]
orb_2_pos = integrator.state_array[:, 6:9]
orb_3_pos = integrator.state_array[:, 12:15]
sun_pos = integrator.state_array[:, 18:21]
venus_pos = integrator.state_array[:, 24:27]
earth_pos = integrator.state_array[:, 30:33]
mars_pos = integrator.state_array[:, 36:39]
jupiter_pos = integrator.state_array[:, 42:45]
saturn_pos = integrator.state_array[:, 48:51]

pos_arrays = [orb_1_pos, orb_2_pos, orb_3_pos, sun_pos, venus_pos, earth_pos, mars_pos, jupiter_pos, saturn_pos]
name_array = ["LEO", "MEO", "GEO", "Sun", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]

pos_arrays = [(pos.transpose(), name) for pos, name in zip(pos_arrays, name_array)]

orb_1_pos_eci = orb_1_pos - earth_pos
orb_2_pos_eci = orb_2_pos - earth_pos
orb_3_pos_eci = orb_3_pos - earth_pos

eci_arrays = [orb_1_pos_eci, orb_2_pos_eci, orb_3_pos_eci]
eci_arrays = [(eci.transpose(), name) for eci, name in zip(eci_arrays, name_array[:3])]

for pos in pos_arrays: ax1.plot(*(pos[0]), '.-', label=pos[1])

for pos in eci_arrays: ax2.plot(*(pos[0]), '.-', label=pos[1])

ax1.legend()

plt.show()


# planets, orbiters = orbits.min_example()
#
# integrator = integrators.BDF2(orbiters, planets, 60)
#
#
# for i in range(N_STEPS):
# 	if i % UPDATE_STEP == 0:
# 		print(f"Running step {i}")
#
# 	integrator.step()
#
# f = plt.figure(figsize=(2000, 2000, 'px'))
#
# ax1 = f.add_subplot(1, 1, 1, projection='3d')
#
# orb_1_pos = integrator.state_array[:, 0:3].transpose()
# orb_2_pos = integrator.state_array[:, 6:9].transpose()
# orb_3_pos = integrator.state_array[:, 12:15].transpose()
#
# ax1.plot(*orb_1_pos, label="Orbiter 1")
# ax1.plot(*orb_2_pos, label="Orbiter 2")
# ax1.plot(*orb_3_pos, label="Orbiter 3")
#
# ax1.legend()
#
# plt.show()

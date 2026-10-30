"""
Handles orbital elements including planets, bodies of negligible mass, and constants

Units are km, s, and kg * 10^24
"""
import numpy as np


_G_mks = 6.67430E-11  # G in si units
G = _G_mks * 1E+15  # G in km, kg^24, and s


class Orbiter:
	"""
	Massless orbiter
	"""
	_taken_ids: int = 0

	def __init__(self, name: str, state: np.ndarray):
		self.name = name
		self.position = np.array(state[0])
		self.velocity = np.array(state[1])
		self.id = Orbiter._taken_ids
		Orbiter._taken_ids += 1


class Planet(Orbiter):
	"""
	Planet of non-negligible mass
	"""
	def __init__(self, name: str, state: np.ndarray, mass: float):
		super().__init__(name, state)
		self.mass: float = mass
		self.grav_param: float = self.mass * G


def min_example() -> tuple[list[Planet], list[Orbiter]]:
	orbiter_names = ["LEO Orbiter", "MEO Orbiter", "GEO Orbiter"]
	r_earth = 6371

	mu_earth = 3.986004418E+05

	# Position and velocity vectors are calculated on an arbitrary plane for now
	r_leo = np.array([r_earth + 1000, 0, 0])  # Halfway to MEO
	r_meo = np.array([r_earth + 20350, 0, 0])  # Around GPS alt
	r_geo = np.array([r_earth + 35786, 0, 0])  # Geostationary orbit

	v_leo = np.array([0, np.sqrt(mu_earth / r_leo[0]), 0])
	v_meo = np.array([0, np.sqrt(mu_earth / r_meo[0]), 0])
	v_geo = np.array([0, np.sqrt(mu_earth / r_geo[0]), 0])

	orbiter_states_rel = [(r_leo, v_leo), (r_meo, v_meo), (r_geo, v_geo)]

	orbiter_states = [np.array(s) for s in orbiter_states_rel]

	# Build orbiters first
	orbs = [Orbiter(*attributes) for attributes in zip(orbiter_names, orbiter_states)]
	planets = [Planet("Earth", np.array([[0, 0, 0], [0, 0, 0]]), 5.97219E+00)]

	orbs.sort(key=lambda x: x.id)
	planets.sort(key=lambda x: x.id)

	return planets, orbs


def get_starting_bodies() -> tuple[list[Planet], list[Orbiter]]:
	planet_names = ["Sun", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]

	# Position elements generated in Horizons (https://ssd.jpl.nasa.gov/horizons/app.html#/)
	# Coordinate system is ICRF with ICRF xy plane, units in km and s, generated for 2026-01-01 @ 00:00:00
	# Structured as arrays of [position; velocity]
	planet_pos_vel = [
		np.array(
			((-4.588639674035421E+05, -7.673041037601740E+05, -3.111955872581376E+05),
			 (1.242505794360611E-02,  3.760600763841905E-04, -9.339934521605677E-05))
		),
		np.array(
			((1.283698332887579E+07, -9.893639739421339E+07, -4.532567953985732E+07),
			 (3.453518141145980E+01,  4.583191517447297E+00, -1.220738584301415E-01))
		),
		np.array(
			((-2.653100241556548E+07,  1.320643995462592E+08,  5.726870337495859E+07),
			 (-2.977650610770464E+01, -4.950768712267514E+00, -2.146229769311948E+00))
		),
		np.array(
			((5.049113049487789E+07, -1.889114225997874E+08, -8.798263151070020E+07),
			 (2.445957741742463E+01,  7.385785743807314E+00,  2.728029472587943E+00))
		),
		np.array(
			((-2.538782093363155E+08,  6.747013995129832E+08,  2.953817193606943E+08),
			 (-1.250761633714600E+01, -3.456931471649910E+00, -1.177327613873146E+00))
		),
		np.array(
			((1.421819203232436E+09,  5.739545228147187E+07, -3.753322772650653E+07),
			 (-7.898651748040556E-01,  8.893137279162939E+00,  3.707923567606738E+00))
		),
	]

	planet_masses = [1988410E+00, 4.8685E+00, 5.97219E+00, 6.4171E-01, 18.9819E+02, 5.6834E+02]

	orbiter_names = ["LEO Orbiter", "MEO Orbiter", "GSO Orbiter"]

	r_earth = 6371
	s_earth = planet_pos_vel[2]
	mu_earth = 3.986004418E+05

	# Position and velocity vectors are calculated on an arbitrary plane for now
	r_leo = np.array([r_earth + 1000, 0, 0])  # Halfway to MEO
	r_meo = np.array([r_earth + 20350, 0, 0])  # Around GPS alt
	r_geo = np.array([r_earth + 35786, 0, 0])  # Geostationary orbit

	v_leo = np.array([0, np.sqrt(mu_earth / r_leo[0]), 0])
	v_meo = np.array([0, np.sqrt(mu_earth / r_meo[0]), 0])
	v_geo = np.array([0, np.sqrt(mu_earth / r_geo[0]), 0])

	orbiter_states_rel = [(r_leo, v_leo), (r_meo, v_meo), (r_geo, v_geo)]

	orbiter_states = [s_earth + np.array(s) for s in orbiter_states_rel]

	# Find barycenter of system, should be slightly off of solar system expected
	bary_coords = sum([rv * m for rv, m in zip(planet_pos_vel, planet_masses)]) / sum(planet_masses)

	orbiter_states = [i - bary_coords for i in orbiter_states]
	planet_pos_vel = [i - bary_coords for i in planet_pos_vel]

	# Build orbiters first
	orbs = [Orbiter(*attributes) for attributes in zip(orbiter_names, orbiter_states)]
	planets = [Planet(*attributes) for attributes in zip(planet_names, planet_pos_vel, planet_masses)]

	orbs.sort(key=lambda x: x.id)
	planets.sort(key=lambda x: x.id)

	return planets, orbs


def f_with_n_orbiters(state: np.ndarray, grav_param_mat: np.ndarray, n_orbiters: int) -> np.ndarray:
	f = np.zeros(state.size)
	n_bodies = len(state) // 6

	for i in range(n_bodies):
		start = 6 * i
		f[start:start + 3] = state[start+3:start+6]

		for j in range(n_bodies - n_orbiters):
			bod_start = 6 * (j + n_orbiters)
			if bod_start == start: continue

			r = state[bod_start:bod_start + 3] - state[start:start + 3]

			f[start + 3:start + 6] += grav_param_mat[j] * r / (np.linalg.norm(r) ** 3)

	return f

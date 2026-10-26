"""
Handles orbital elements including planets, bodies of negligible mass, and constants

Units are km, s, and kg * 10^24
"""
import numpy as np


_G_mks = 6.67430E-11  # G in si units
G = _G_mks * 1E-9 * 1E-24  # G in km, kg^24, and s


class Orbiter:
	"""
	Massless orbiter
	"""
	def __init__(self, name: str, state: np.ndarray):
		self.name = name
		self.position = np.array(state[0])
		self.velocity = np.array(state[1])


class Planet:
	"""
	Planet of non-negligible mass
	"""
	def __init__(self, name: str, state: np.ndarray, mass: float):
		self.name: str = name
		self.position: np.ndarray = state[0]
		self.velocity: np.ndarray = state[1]
		self.mass: float = mass


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

	orbiter_names = ["LEO Orbiter", "MEO Orbiter", "GEO Orbiter"]

	r_earth = 6371
	s_earth = planet_pos_vel[2]
	mu_earth = 3.986004418E+05

	# Position and velocity vectors are calculated on an arbitrary plane for now
	r_leo = np.array([r_earth + 1000, 0, 0])  # Halfway to MEO
	r_meo = np.array([r_earth + 20350, 0, 0])  # Around GPS alt
	r_geo = np.array([r_earth + 35786, 0, 0])  # Geostationary orbit

	v_leo = np.array([0, np.sqrt(mu_earth / r_leo), 0])
	v_meo = np.array([0, np.sqrt(mu_earth / r_meo), 0])
	v_geo = np.array([0, np.sqrt(mu_earth / r_geo), 0])

	orbiter_states_rel = [(r_leo, v_leo), (r_meo, v_meo), (r_geo, v_geo)]

	orbiter_states = [s_earth + np.array(s) for s in orbiter_states_rel]

	return (
		[Planet(*attributes) for attributes in zip(planet_names, planet_pos_vel, planet_masses)],
		[Orbiter(*attributes) for attributes in zip(orbiter_names, orbiter_states)]
	)

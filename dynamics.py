import numpy as np
import sympy as sp
import orbits

def bary_shift(x, mu):
	X = x.reshape(-1, 6).copy()
	X[:, :3] -= mu @ X[:, :3] / mu.sum()
	X[:, 3:] -= mu @ X[:, 3:] / mu.sum()
	
	return X.ravel()


def build(planets, orbiters):
	bodies = list(planets) + list(orbiters)
	x = np.concatenate([np.r_[b.position, b.velocity] for b in bodies]).astype(float)
	
	mu = np.array([orbits.G*p.mass for p in planets] + [0.0]*len(orbiters))
	
	return x, mu, [b.name for b in bodies]


def rhs(x, mu):
	X = x.reshape(-1, 6)
	r, v = X[:, :3], X[:, 3:]

	d = r[None, :, :] - r[:, None, :]
	dist = np.linalg.norm(d, axis=2)
	np.fill_diagonal(dist, np.inf)
	a = np.sum(mu[None, :, None] * d / dist[:, :, None] ** 3, axis=1)

	x_dot = np.hstack([v, a]).ravel()
	
	return x_dot
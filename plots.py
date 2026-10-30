"""
Figures for the analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import methods


def stability_regions(lam, dts, label="", which=None, xlim=(-4.0, 3.5), ylim=(-3.5, 3.5), n_grid=401):
	'''Regions of absolute stability, one panel per method, with w = dt * lambda marked for each dt.

	Parameters
	----------
	lam : ndarray of complex, the eigenvalues to mark
	dts : list of float, step sizes (s); each gets its own marker
	label : str, prefix for the legend entries (e.g. "LEO" or "all modes")
	which : list of method names from methods.METHODS (default: all of them)

	Returns
	-------
	fig : the matplotlib figure (call plt.show() or fig.savefig after)
	'''
	which = list(methods.METHODS) if which is None else which
	x, y = np.meshgrid(np.linspace(*xlim, n_grid), np.linspace(*ylim, n_grid))
	W = x + 1j * y
	markers = ['x', 'o', 's', '^']

	fig, axes = plt.subplots(1, len(which), figsize=(4 * len(which), 3.6), sharey=True)
	axes = np.atleast_1d(axes)
	for ax, name in zip(axes, which):
		amp = methods.METHODS[name]["amp"](W)
		ax.contourf(x, y, amp <= 1, levels=[0.5, 1.5], colors='orange')
		ax.contour(x, y, amp, levels=[1.0], colors='k', linewidths=0.7)
		for dt, mk in zip(dts, markers):
			w = dt * lam
			ax.plot(w.real, w.imag, mk, color='navy', mfc='none', label=f"{label} dt={dt:.0f} s".strip())
		ax.axhline(0, color='k', lw=0.4)
		ax.axvline(0, color='k', lw=0.4)
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		ax.set_title(name)
		ax.set_xlabel(r"Re $w$")
	axes[0].set_ylabel(r"Im $w$")
	axes[0].legend(fontsize=7, loc='lower left')
	fig.tight_layout()
	return fig

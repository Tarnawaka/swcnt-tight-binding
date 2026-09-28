"""Density of states from band energies sampled on a uniform k-grid."""
from __future__ import annotations

import numpy as np


def broadened_dos(energies, norm, emin, emax, sigma=0.02, bin_width=None):
    """Histogram band energies and smooth the result with a Gaussian.

    Parameters
    ----------
    energies : array_like
        Band energies (eV) sampled on a uniform k-point grid.
    norm : float
        Number of k-points times the number of atoms per unit cell. Dividing
        by it gives states / eV / atom for one spin channel, so the DOS of the
        full pi-band manifold integrates to 1.
    emin, emax : float
        Energy window (eV).
    sigma : float
        Standard deviation of the Gaussian broadening (eV).
    bin_width : float, optional
        Histogram bin width (eV); defaults to sigma / 5.

    Returns
    -------
    energy_grid, dos : ndarray
    """
    if bin_width is None:
        bin_width = sigma / 5 if sigma > 0 else 1e-3
    pad = 5 * sigma
    edges = np.arange(emin - pad, emax + pad + bin_width, bin_width)
    counts, edges = np.histogram(np.ravel(energies), bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dos = counts / (norm * bin_width)
    if sigma > 0:
        half = int(np.ceil(4 * sigma / bin_width))
        x = bin_width * np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        dos = np.convolve(dos, kernel / kernel.sum(), mode="same")
    keep = (centers >= emin) & (centers <= emax)
    return centers[keep], dos[keep]

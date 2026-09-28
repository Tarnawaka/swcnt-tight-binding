"""Kataura plot data and laser-resonance search across many chiralities."""
from __future__ import annotations

import numpy as np

from . import graphene
from .nanotube import Nanotube, Transition
from .raman import RBM_A, RBM_B, laser_energy, rbm_frequency


def _diameter(n: int, m: int) -> float:
    return graphene.LATTICE_CONSTANT / 10 * np.sqrt(n * n + m * m + n * m) / np.pi


def chiralities(dmin: float, dmax: float) -> list[tuple[int, int]]:
    """All (n, m) with n >= m >= 0 and dmin <= diameter <= dmax (nm), thinnest first."""
    n_max = int(np.pi * dmax / (graphene.LATTICE_CONSTANT / 10)) + 1
    found = [(n, m) for n in range(1, n_max + 1) for m in range(n + 1)
             if dmin <= _diameter(n, m) <= dmax]
    return sorted(found, key=lambda nm: _diameter(*nm))


def kataura_data(dmin=0.6, dmax=2.0, emax=3.5, gamma0=graphene.GAMMA0, overlap=0.0):
    """List of (tube, transitions) for every chirality in the diameter range."""
    data = []
    for n, m in chiralities(dmin, dmax):
        tube = Nanotube(n, m, gamma0, overlap)
        data.append((tube, tube.transition_energies(emax=emax)))
    return data


def resonant_tubes(wavelength_nm, window=0.1, dmin=0.6, dmax=2.0, gamma0=graphene.GAMMA0,
                   overlap=0.0, rbm=None, rbm_tolerance=5.0, rbm_A=RBM_A, rbm_B=RBM_B
                   ) -> list[tuple[Nanotube, Transition]]:
    """Tubes with a transition E_ii within +/- window (eV) of the laser energy.

    If rbm (cm^-1) is given, only tubes whose predicted RBM frequency lies
    within rbm_tolerance of it are kept, which mimics assigning an observed
    RBM peak to candidate (n, m) species.
    """
    e_laser = laser_energy(wavelength_nm)
    hits = []
    for n, m in chiralities(dmin, dmax):
        tube = Nanotube(n, m, gamma0, overlap)
        if rbm is not None and abs(rbm_frequency(tube.diameter, rbm_A, rbm_B) - rbm) > rbm_tolerance:
            continue
        for t in tube.transition_energies(emax=e_laser + window + 0.01):
            if abs(t.energy - e_laser) <= window:
                hits.append((tube, t))
    return hits

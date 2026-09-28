"""
Nearest-neighbour tight-binding model of graphene's pi bands.

Each carbon atom contributes one 2p_z orbital, and the honeycomb lattice has
two atoms (A and B) per unit cell, so the Bloch Hamiltonian H(k) and the
overlap matrix S(k) are 2 x 2:

    H(k) = |  eps            -gamma0 f(k) |      S(k) = |  1         s f(k) |
           | -gamma0 f*(k)    eps         |             |  s f*(k)   1      |

The structure factor f(k) = 1 + exp(i k.a1) + exp(i k.a2) sums the phases of
the three nearest neighbours. Solving det(H - E S) = 0 gives

    E_c(k) = (eps + gamma0 w) / (1 - s w)     conduction band (pi*)
    E_v(k) = (eps - gamma0 w) / (1 + s w)     valence band (pi)

with w(k) = |f(k)|. For s = 0 the bands are symmetric, E = +/- gamma0 w, and
they touch (w = 0) at the corners K and K' of the Brillouin zone: the Dirac
points.

Units: lengths in angstrom, wavevectors in 1/angstrom, energies in eV.
Conventions follow Saito, Dresselhaus & Dresselhaus, "Physical Properties of
Carbon Nanotubes" (1998).
"""
from __future__ import annotations

import numpy as np

from .dos import broadened_dos

A_CC = 1.42                            # carbon-carbon bond length (angstrom)
LATTICE_CONSTANT = np.sqrt(3) * A_CC   # |a1| = |a2|, about 2.46 angstrom
GAMMA0 = 2.7                           # nearest-neighbour hopping energy (eV)

_a = LATTICE_CONSTANT
A1 = _a * np.array([np.sqrt(3) / 2, 0.5])    # real-space lattice vectors
A2 = _a * np.array([np.sqrt(3) / 2, -0.5])

# Reciprocal lattice vectors, defined by a_i . b_j = 2 pi delta_ij
B1, B2 = 2 * np.pi * np.linalg.inv(np.array([A1, A2])).T

# High-symmetry points of the hexagonal Brillouin zone
GAMMA = np.zeros(2)
M_POINT = (B1 + B2) / 2          # middle of a zone edge (saddle point)
K_POINT = (2 * B1 + B2) / 3      # zone corner (Dirac point)
K_PRIME = (B1 + 2 * B2) / 3      # neighbouring, inequivalent zone corner


def structure_factor_sq(kx, ky):
    """|f(k)|^2 = 3 + 2 cos(k.a1) + 2 cos(k.a2) + 2 cos(k.(a1 - a2))."""
    t1 = kx * A1[0] + ky * A1[1]
    t2 = kx * A2[0] + ky * A2[1]
    w2 = 3 + 2 * np.cos(t1) + 2 * np.cos(t2) + 2 * np.cos(t1 - t2)
    return np.clip(w2, 0.0, None)   # remove tiny negative rounding errors


def band_energies(w, gamma0=GAMMA0, overlap=0.0, eps=0.0):
    """Valence and conduction energies for given values of w = |f(k)|."""
    w = np.asarray(w, dtype=float)
    ev = (eps - gamma0 * w) / (1 + overlap * w)
    ec = (eps + gamma0 * w) / (1 - overlap * w)
    return ev, ec


def bands(kx, ky, gamma0=GAMMA0, overlap=0.0):
    """Graphene's pi (valence) and pi* (conduction) energies at (kx, ky)."""
    return band_energies(np.sqrt(structure_factor_sq(kx, ky)), gamma0, overlap)


def hamiltonian(k, gamma0=GAMMA0, overlap=0.0, eps=0.0):
    """The 2 x 2 matrices H(k) and S(k) at a single wavevector k.

    The closed-form bands above are derived from these; the tests check the
    two against each other by solving H psi = E S psi numerically.
    """
    f = 1 + np.exp(1j * np.dot(k, A1)) + np.exp(1j * np.dot(k, A2))
    H = np.array([[eps, -gamma0 * f], [-gamma0 * np.conj(f), eps]])
    S = np.array([[1, overlap * f], [overlap * np.conj(f), 1]])
    return H, S


def high_symmetry_path(density=150):
    """k-points along Gamma -> M -> K -> Gamma.

    Returns the k-points (n x 2), the distance travelled along the path, the
    distances of the corner points (for axis ticks), and their labels.
    """
    nodes = [GAMMA, M_POINT, K_POINT, GAMMA]
    labels = [r"$\Gamma$", "M", "K", r"$\Gamma$"]
    kpts, dist, ticks, total = [], [], [0.0], 0.0
    for start, end in zip(nodes[:-1], nodes[1:]):
        length = np.linalg.norm(end - start)
        t = np.linspace(0, 1, max(2, int(density * length)), endpoint=False)
        kpts.append(start + t[:, None] * (end - start))
        dist.append(total + t * length)
        total += length
        ticks.append(total)
    kpts.append(nodes[-1][None, :])
    dist.append(np.array([total]))
    return np.vstack(kpts), np.concatenate(dist), np.array(ticks), labels


def density_of_states(emin=-9.0, emax=9.0, sigma=0.05, nk=1000, gamma0=GAMMA0, overlap=0.0):
    """Graphene DOS in states / eV / atom (one spin), from an nk x nk k-grid."""
    u = np.arange(nk) / nk
    U, V = np.meshgrid(u, u, indexing="ij")
    ev, ec = bands(U * B1[0] + V * B2[0], U * B1[1] + V * B2[1], gamma0, overlap)
    energies = np.concatenate([ev.ravel(), ec.ravel()])
    return broadened_dos(energies, norm=2 * nk * nk, emin=emin, emax=emax, sigma=sigma)

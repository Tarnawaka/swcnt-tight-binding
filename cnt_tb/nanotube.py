"""
Zone-folding tight-binding model of single-walled carbon nanotubes (SWCNTs).

A nanotube with chiral indices (n, m) is a graphene sheet rolled up along the
chiral vector Ch = n a1 + m a2. Going once around the circumference must bring
the electron wavefunction back to itself, so the wavevector component along Ch
is quantised. Only states on N parallel "cutting lines" through graphene's
Brillouin zone survive:

    k = mu K1 + k_par K2 / |K2|,    mu = 0, ..., N - 1,    -pi/T <= k_par < pi/T

K1 points around the circumference (Ch . K1 = 2 pi) and K2 along the tube axis
(T . K2 = 2 pi), where T is the translation vector of the 1D unit cell. Each
nanotube band is graphene's band evaluated along one cutting line.

Notation follows Saito, Dresselhaus & Dresselhaus (1998), chapter 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np

from . import graphene
from .dos import broadened_dos
from .raman import photon_wavelength, rbm_frequency


@dataclass(frozen=True)
class Transition:
    """An optical transition E_ii between mirror-image van Hove singularities."""

    index: int      # i in E_ii
    family: str     # "S" (semiconducting) or "M" (metallic)
    branch: str     # "", "-" or "+" (trigonal-warping split of metallic peaks)
    energy: float   # transition energy E_c - E_v (eV)
    ev: float       # valence-band singularity (eV)
    ec: float       # conduction-band singularity (eV)

    @property
    def _ii(self) -> str:
        return f"{self.index}{self.index}" if self.index < 10 else f"{self.index},{self.index}"

    @property
    def label(self) -> str:
        """Plain-text label such as 'E22S' or 'E11M-'."""
        return f"E{self._ii}{self.family}{self.branch}"

    @property
    def tex(self) -> str:
        """Matplotlib label such as $E_{22}^{S}$."""
        return rf"$E_{{{self._ii}}}^{{{self.family}{self.branch}}}$"

    @property
    def wavelength(self) -> float:
        """Photon wavelength (nm) that matches this transition."""
        return photon_wavelength(self.energy)


class Nanotube:
    """A single-walled carbon nanotube with chiral indices (n, m).

    Parameters
    ----------
    n, m : int
        Chiral indices. (m, n) is the mirror image of (n, m) and has the same
        band structure, so the indices are ordered so that n >= m.
    gamma0 : float
        Nearest-neighbour hopping energy (eV).
    overlap : float
        Overlap integral s between neighbouring orbitals (0 gives symmetric bands).
    """

    def __init__(self, n: int, m: int, gamma0: float = graphene.GAMMA0, overlap: float = 0.0):
        n, m = int(n), int(m)
        if n < 0 or m < 0 or n + m == 0:
            raise ValueError("Chiral indices must be non-negative and not both zero.")
        if m > n:
            n, m = m, n
        self.n, self.m = n, m
        self.gamma0 = gamma0
        self.overlap = overlap

        self.d_R = gcd(2 * n + m, 2 * m + n)
        self.N = 2 * (n * n + m * m + n * m) // self.d_R    # hexagons per unit cell
        self.t1 = (2 * m + n) // self.d_R
        self.t2 = -((2 * n + m) // self.d_R)

        a1, a2, b1, b2 = graphene.A1, graphene.A2, graphene.B1, graphene.B2
        self.Ch = n * a1 + m * a2                            # chiral vector
        self.T = self.t1 * a1 + self.t2 * a2                 # translation vector
        self.K1 = (-self.t2 * b1 + self.t1 * b2) / self.N    # around the tube (quantised)
        self.K2 = (m * b1 - n * b2) / self.N                 # along the axis (continuous)
        self._axis = self.K2 / np.linalg.norm(self.K2)

    def __repr__(self) -> str:
        return f"Nanotube({self.n}, {self.m})"

    # ------------------------------------------------------------------ geometry
    @property
    def name(self) -> str:
        return f"({self.n},{self.m})"

    @property
    def kind(self) -> str:
        if self.n == self.m:
            return "armchair"
        return "zigzag" if self.m == 0 else "chiral"

    @property
    def diameter(self) -> float:
        """Tube diameter (nm)."""
        return np.linalg.norm(self.Ch) / np.pi / 10

    @property
    def chiral_angle(self) -> float:
        """Angle between Ch and a1 in degrees: 0 for zigzag, 30 for armchair."""
        return float(np.degrees(np.arctan2(np.sqrt(3) * self.m, 2 * self.n + self.m)))

    @property
    def unit_cell_length(self) -> float:
        """Length |T| of the 1D unit cell (angstrom)."""
        return float(np.linalg.norm(self.T))

    @property
    def num_atoms(self) -> int:
        """Carbon atoms per unit cell."""
        return 2 * self.N

    @property
    def is_metallic(self) -> bool:
        """A cutting line passes through K exactly when (n - m) is divisible by 3."""
        return (self.n - self.m) % 3 == 0

    @property
    def k_max(self) -> float:
        """Edge of the 1D Brillouin zone, pi / |T| (1/angstrom)."""
        return np.pi / self.unit_cell_length

    # ------------------------------------------------------------- electronics
    def line_points(self, mu, k):
        """Graphene wavevectors (kx, ky) at position k along cutting line mu."""
        kx = mu * self.K1[0] + k * self._axis[0]
        ky = mu * self.K1[1] + k * self._axis[1]
        return kx, ky

    def _w2(self, mu, k):
        return graphene.structure_factor_sq(*self.line_points(mu, k))

    def _energies_on_lines(self, k):
        w = np.sqrt(self._w2(np.arange(self.N)[:, None], k[None, :]))
        return graphene.band_energies(w, self.gamma0, self.overlap)

    def bands(self, nk=None):
        """1D band structure.

        Returns k (1/angstrom) and the valence and conduction energies, each an
        array of shape (N, nk): one band per cutting line.
        """
        if nk is None:
            nk = int(np.clip(2 * self.k_max / 0.005, 51, 401))
        k = np.linspace(-self.k_max, self.k_max, nk)
        ev, ec = self._energies_on_lines(k)
        return k, ev, ec

    def subband_minima(self, nk=241, margin=0.2, refinements=3):
        """Locate the band edges (local minima of |f|) on every cutting line.

        Returns (mu, k, w): the line index, the position along the line, and
        the value of |f| at each minimum. The conduction band has a minimum and
        the valence band a maximum there, so dE/dk = 0 and the density of
        states has a van Hove singularity.

        Each line is sampled a little beyond the 1D zone, so that minima on the
        zone boundary still show up as interior minima. A minimum is kept only
        if it lies in [-pi/T, pi/T): the part of a line past the boundary holds
        the same states as the start of another line, so nothing is counted
        twice. Each minimum is refined by repeated parabolic interpolation.
        """
        kmax = self.k_max
        k = np.linspace(-(1 + margin) * kmax, (1 + margin) * kmax, nk)
        h = k[1] - k[0]
        w2 = self._w2(np.arange(self.N)[:, None], k[None, :])
        left, mid, right = w2[:, :-2], w2[:, 1:-1], w2[:, 2:]
        mu, i = np.nonzero((mid < left) & (mid <= right))
        k_min = k[i + 1]
        l, c, r = left[mu, i], mid[mu, i], right[mu, i]
        for step in range(refinements + 1):
            if step > 0:
                h /= 8
                l, c, r = self._w2(mu, k_min - h), self._w2(mu, k_min), self._w2(mu, k_min + h)
            curvature = l - 2 * c + r
            ok = curvature > 0
            safe = np.where(ok, curvature, 1.0)
            k_min = k_min + np.where(ok, 0.5 * (l - r) / safe, 0.0) * h
            w2_min = np.where(ok, c - 0.125 * (l - r) ** 2 / safe, c)
        inside = (k_min >= -kmax) & (k_min < kmax)
        return mu[inside], k_min[inside], np.sqrt(np.clip(w2_min[inside], 0.0, None))

    def band_gap(self) -> float:
        """Band gap (eV); zero for metallic tubes."""
        _, _, w = self.subband_minima()
        ev, ec = graphene.band_energies(w.min(), self.gamma0, self.overlap)
        return max(float(ec - ev), 0.0)

    def transition_energies(self, emax: float = 4.0, tol: float = 1e-4) -> list[Transition]:
        """Optical transitions E_ii up to emax (eV), lowest first.

        E_ii connects the i-th valence and conduction van Hove singularities.
        Every value appears twice (from the K and K' valleys), so duplicates
        are merged. In metallic tubes that are not armchair, trigonal warping
        splits each E_ii^M into a lower (-) and an upper (+) branch.
        """
        _, _, w = self.subband_minima()
        ev, ec = graphene.band_energies(w, self.gamma0, self.overlap)
        values = []
        for j in np.argsort(ec - ev):
            e = float(ec[j] - ev[j])
            if e < 1e-3 or e > emax:        # skip the metallic band crossing
                continue
            if values and e - values[-1][0] < tol:
                continue
            values.append((e, float(ev[j]), float(ec[j])))

        family = "M" if self.is_metallic else "S"
        split = self.is_metallic and self.kind != "armchair"
        transitions = []
        for j, (e, lo, hi) in enumerate(values):
            index, branch = (j // 2 + 1, "-+"[j % 2]) if split else (j + 1, "")
            transitions.append(Transition(index, family, branch, e, lo, hi))
        return transitions

    def density_of_states(self, emin=-3.0, emax=3.0, sigma=0.02, dk=5e-4):
        """DOS in states / eV / atom (one spin), Gaussian-broadened by sigma (eV)."""
        nk = max(64, int(np.ceil(2 * self.k_max / dk)))
        k = np.linspace(-self.k_max, self.k_max, nk, endpoint=False)
        ev, ec = self._energies_on_lines(k)
        energies = np.concatenate([ev.ravel(), ec.ravel()])
        return broadened_dos(energies, norm=nk * self.num_atoms, emin=emin, emax=emax, sigma=sigma)

    # ----------------------------------------------------------------- summary
    def summary(self) -> str:
        """Readable overview of the tube's geometry and electronic structure."""
        etype = "metallic" if self.is_metallic else "semiconducting"
        lines = [
            f"{self.name} {self.kind} SWCNT",
            f"  diameter            {self.diameter:.3f} nm",
            f"  chiral angle        {self.chiral_angle:.2f} deg",
            f"  electronic type     {etype}  ((n - m) mod 3 = {(self.n - self.m) % 3})",
            f"  hexagons / cell     {self.N}",
            f"  atoms / cell        {self.num_atoms}",
            f"  cell length |T|     {self.unit_cell_length:.3f} angstrom",
            f"  band gap            {self.band_gap():.3f} eV",
            f"  RBM (248/d)         {rbm_frequency(self.diameter):.1f} cm^-1",
            f"  optical transitions (gamma0 = {self.gamma0:.2f} eV):",
        ]
        for t in self.transition_energies():
            lines.append(f"    {t.label:<8}{t.energy:6.3f} eV   ({t.wavelength:5.0f} nm)")
        return "\n".join(lines)

"""Figures: graphene bands, cutting lines, nanotube bands and DOS, Kataura plot."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from . import graphene
from .kataura import kataura_data
from .raman import RBM_A, RBM_B, laser_energy

BAND_COLOR = "#1f4e79"
SEMI_COLOR = "#1f77b4"
METAL_COLOR = "#d62728"
LINE_COLOR = "#e8590c"
LASER_COLORS = {488: "#17becf", 514: "#17becf", 532: "#2ca02c", 633: "#e377c2", 785: "#8c6bb1"}
STYLE = {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10}


# ---------------------------------------------------------------- helpers
def _bz_corners():
    """The six corners (K and K' points) of graphene's hexagonal Brillouin zone."""
    r = np.linalg.norm(graphene.K_POINT)
    angles = np.radians(30 + 60 * np.arange(6))
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def fold_to_first_bz(kx, ky):
    """Translate wavevectors by reciprocal lattice vectors into the first zone."""
    k = np.stack([np.asarray(kx, float), np.asarray(ky, float)], axis=-1)
    basis = np.array([graphene.B1, graphene.B2])
    frac = k @ np.linalg.inv(basis)
    base = (frac - np.floor(frac)) @ basis          # inside the b1-b2 rhombus
    best, best_d = base, np.linalg.norm(base, axis=-1)
    for i in (0, 1):                                 # nearest rhombus corner is the
        for j in (0, 1):                             # lattice point to subtract
            cand = base - (i * graphene.B1 + j * graphene.B2)
            d = np.linalg.norm(cand, axis=-1)
            better = d < best_d
            best = np.where(better[..., None], cand, best)
            best_d = np.where(better, d, best_d)
    return best[..., 0], best[..., 1]


# ---------------------------------------------------------------- graphene
def plot_graphene(gamma0=graphene.GAMMA0, overlap=0.0):
    """Band structure along Gamma-M-K-Gamma, DOS, and a 3D view of the Dirac cones."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 4.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.55, 1.35], wspace=0.12)

        ax1 = fig.add_subplot(gs[0])
        kpts, dist, ticks, labels = graphene.high_symmetry_path()
        ev, ec = graphene.bands(kpts[:, 0], kpts[:, 1], gamma0, overlap)
        ax1.plot(dist, ev, color=BAND_COLOR, lw=1.6)
        ax1.plot(dist, ec, color=BAND_COLOR, lw=1.6)
        for t in ticks:
            ax1.axvline(t, color="0.75", lw=0.8)
        ax1.axhline(0, color="0.5", ls="--", lw=0.8)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels)
        ax1.set_xlim(dist[0], dist[-1])
        elo, ehi = ev.min() - 0.5, ec.max() + 0.5
        ax1.set_ylim(elo, ehi)
        ax1.set_ylabel("Energy (eV)")
        ax1.set_title("Band structure")

        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        E, g = graphene.density_of_states(elo, ehi, gamma0=gamma0, overlap=overlap)
        ax2.fill_betweenx(E, 0, g, color=BAND_COLOR, alpha=0.25, lw=0)
        ax2.plot(g, E, color=BAND_COLOR, lw=1.0)
        ax2.axhline(0, color="0.5", ls="--", lw=0.8)
        ax2.set_xlim(0, 1.05 * g.max())
        ax2.set_xlabel("DOS (states/eV/atom)")
        ax2.set_title("Density of states")
        plt.setp(ax2.get_yticklabels(), visible=False)

        ax3 = fig.add_subplot(gs[2], projection="3d")
        r = 1.1 * np.linalg.norm(graphene.K_POINT)
        grid = np.linspace(-r, r, 121)
        kx, ky = np.meshgrid(grid, grid)
        ev3, ec3 = graphene.bands(kx, ky, gamma0, overlap)
        norm = plt.Normalize(ev3.min(), ec3.max())
        for surface in (ev3, ec3):
            ax3.plot_surface(kx, ky, surface, cmap="coolwarm", norm=norm, rstride=2,
                             cstride=2, linewidth=0, antialiased=True)
        ax3.set_xlabel(r"$k_x$ (Å$^{-1}$)", labelpad=6)
        ax3.set_ylabel(r"$k_y$ (Å$^{-1}$)", labelpad=6)
        ax3.set_zlabel("E (eV)", labelpad=4)
        ax3.set_xticks([-1.5, 0, 1.5])
        ax3.set_yticks([-1.5, 0, 1.5])
        ax3.set_zticks([-8, -4, 0, 4, 8])
        ax3.view_init(elev=16, azim=-58)
        try:
            ax3.set_box_aspect(None, zoom=0.85)   # leave room for the z label
        except TypeError:                          # matplotlib < 3.6
            pass
        ax3.set_title("Dirac cones at the zone corners")
        fig.suptitle(f"Graphene, nearest-neighbour tight binding ($\\gamma_0$ = {gamma0:g} eV)",
                     y=1.02, fontsize=13)
    return fig


# ---------------------------------------------------------------- nanotubes
def plot_cutting_lines(tube, ax=None, background=True):
    """Graphene's Brillouin zone with the tube's allowed states (cutting lines)."""
    ax = ax if ax is not None else plt.gca()
    corners = _bz_corners()
    r = 1.12 * np.linalg.norm(graphene.K_POINT)
    if background:
        grid = np.linspace(-r, r, 301)
        kx, ky = np.meshgrid(grid, grid)
        _, ec = graphene.bands(kx, ky, tube.gamma0, tube.overlap)
        ax.contourf(kx, ky, ec, levels=20, cmap="Blues")
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="k", lw=1.2))

    k = np.linspace(-tube.k_max, tube.k_max, max(20, int(2 * tube.k_max / 0.004) + 1))
    segments = []
    for mu in range(tube.N):
        pts = np.column_stack(fold_to_first_bz(*tube.line_points(mu, k)))
        breaks = np.nonzero(np.hypot(*np.diff(pts, axis=0).T) > 0.1)[0] + 1
        segments += [p for p in np.split(pts, breaks) if len(p) > 1]
    ax.add_collection(LineCollection(segments, colors=LINE_COLOR, linewidths=1.4))

    ax.plot(corners[:, 0], corners[:, 1], "o", ms=6.5, mfc="k", mec="w", mew=1.0, zorder=4)
    for point, label in ((graphene.K_POINT, "K"), (graphene.K_PRIME, "K′")):
        ax.annotate(label, point, xytext=(7, 0), textcoords="offset points", va="center")
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$ (Å$^{-1}$)")
    ax.set_ylabel(r"$k_y$ (Å$^{-1}$)")
    return ax


def plot_bands(tube, ax=None, emax=2.5):
    """All 1D bands within +/- emax of the Fermi level."""
    ax = ax if ax is not None else plt.gca()
    k, ev, ec = tube.bands()
    x = k / tube.k_max
    segments = [np.column_stack([x, e]) for e in np.vstack([ev, ec])
                if e.min() < emax and e.max() > -emax]
    ax.add_collection(LineCollection(segments, colors=BAND_COLOR, linewidths=0.9))
    ax.axhline(0, color="0.5", ls="--", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-emax, emax)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(r"$k$ ($\pi/T$)")
    ax.set_ylabel("Energy (eV)")
    return ax


def plot_dos(tube, ax=None, emax=2.5, sigma=0.02, n_arrows=3):
    """DOS with arrows marking the first optical transitions E_ii."""
    ax = ax if ax is not None else plt.gca()
    E, g = tube.density_of_states(-emax, emax, sigma=sigma)
    ax.fill_betweenx(E, 0, g, color=BAND_COLOR, alpha=0.25, lw=0)
    ax.plot(g, E, color=BAND_COLOR, lw=0.9)
    ax.axhline(0, color="0.5", ls="--", lw=0.8)
    gmax = 1.05 * g.max()
    ax.set_xlim(0, gmax)
    ax.set_ylim(-emax, emax)
    ax.set_xlabel("DOS (states/eV/atom)")

    shown = [t for t in tube.transition_energies() if t.ec < emax - 0.3][:n_arrows]
    for j, t in enumerate(shown):
        x = gmax * (0.58 + 0.15 * j)
        ax.annotate("", xy=(x, t.ec), xytext=(x, t.ev),
                    arrowprops=dict(arrowstyle="<->", color="0.15", lw=1.1, shrinkA=0, shrinkB=0))
        ax.text(x, t.ec + 0.06, t.tex, ha="center", va="bottom", fontsize=10)
    return ax


def plot_tube(tube, emax=2.5, sigma=0.02):
    """Three panels: cutting lines, band structure, and density of states."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 4.9))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.7], wspace=0.25)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2], sharey=ax1)

        plot_cutting_lines(tube, ax0)
        ax0.set_title("Cutting lines in graphene's zone")
        plot_bands(tube, ax1, emax)
        ax1.set_title(f"Band structure ({2 * tube.N} bands)")
        plot_dos(tube, ax2, emax, sigma)
        ax2.set_title("Density of states")
        ax2.set_ylabel("")
        plt.setp(ax2.get_yticklabels(), visible=False)

        etype = "metallic" if tube.is_metallic else f"semiconducting, $E_g$ = {tube.band_gap():.2f} eV"
        fig.suptitle(f"{tube.name} {tube.kind} SWCNT:  d = {tube.diameter:.3f} nm,  "
                     f"θ = {tube.chiral_angle:.1f}°,  {etype}", y=1.02, fontsize=13)
    return fig


def plot_cutting_line_comparison(tubes):
    """Side-by-side cutting-line plots, e.g. a metallic and a semiconducting tube."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(tubes), figsize=(5.2 * len(tubes), 5.2))
        for ax, tube in zip(np.atleast_1d(axes), tubes):
            plot_cutting_lines(tube, ax)
            etype = ("metallic: lines pass through K" if tube.is_metallic
                     else "semiconducting: lines miss K")
            ax.set_title(f"{tube.name}, d = {tube.diameter:.2f} nm\n{etype}")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Kataura plot
def plot_kataura(dmin=0.6, dmax=2.0, emax=3.5, lasers_nm=(532, 633, 785), window=0.1,
                 gamma0=graphene.GAMMA0, overlap=0.0, rbm_A=RBM_A, rbm_B=RBM_B):
    """Transition energies E_ii versus diameter, with Raman laser resonance windows."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        for lam in lasers_nm:
            e = laser_energy(lam)
            color = LASER_COLORS.get(int(round(lam)), "0.4")
            ax.axhspan(e - window, e + window, color=color, alpha=0.15, lw=0)
            ax.axhline(e, color=color, lw=1.0)
            ax.text(1.01, e, f"{lam:g} nm", color=color, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=10, fontweight="bold")

        d = np.linspace(dmin, dmax, 300)
        a_cc_nm = graphene.A_CC / 10
        for p, color in ((2, SEMI_COLOR), (4, SEMI_COLOR), (6, METAL_COLOR),
                         (8, SEMI_COLOR), (10, SEMI_COLOR), (12, METAL_COLOR)):
            ax.plot(d, p * a_cc_nm * gamma0 / d, ls="--", lw=0.8, color=color, alpha=0.45)

        ds, es, dm, em = [], [], [], []
        for tube, transitions in kataura_data(dmin, dmax, emax, gamma0, overlap):
            for t in transitions:
                (dm if t.family == "M" else ds).append(tube.diameter)
                (em if t.family == "M" else es).append(t.energy)
        ax.scatter(ds, es, s=13, color=SEMI_COLOR, label="semiconducting", zorder=3)
        ax.scatter(dm, em, s=18, marker="^", color=METAL_COLOR, label="metallic", zorder=3)
        ax.plot([], [], ls="--", color="0.5", label="linear-dispersion estimate")

        ax.set_xlim(dmin, dmax)
        ax.set_ylim(0, emax)
        ax.set_xlabel("Tube diameter d (nm)")
        ax.set_ylabel(r"Transition energy $E_{ii}$ (eV)")
        ax.legend(loc="lower left", framealpha=0.95)

        def to_rbm(x):
            x = np.asarray(x, float)
            return rbm_A / np.where(np.abs(x) < 1e-6, 1e-6, x) + rbm_B

        def from_rbm(w):
            w = np.asarray(w, float)
            return rbm_A / np.where(np.abs(w - rbm_B) < 1e-6, 1e-6, w - rbm_B)

        top = ax.secondary_xaxis("top", functions=(to_rbm, from_rbm))
        top.set_xlabel(f"RBM frequency, {rbm_A:g}/d" + (f" + {rbm_B:g}" if rbm_B else "")
                       + r" (cm$^{-1}$)")
        ax.set_title(f"Kataura plot (nearest-neighbour tight binding, $\\gamma_0$ = {gamma0:g} eV)",
                     pad=12)
        fig.tight_layout()
    return fig

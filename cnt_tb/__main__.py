"""Command-line interface.

    python -m cnt_tb tube 10 0            summary and figure for one tube
    python -m cnt_tb laser 532            tubes resonant with a 532 nm laser
    python -m cnt_tb laser 532 --rbm 165  ... that also match an RBM peak
    python -m cnt_tb figures              regenerate every README figure
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

from . import plotting  # noqa: E402
from .graphene import GAMMA0  # noqa: E402
from .kataura import resonant_tubes  # noqa: E402
from .nanotube import Nanotube  # noqa: E402
from .raman import RBM_A, RBM_B, laser_energy, rbm_frequency  # noqa: E402


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plotting.plt.close(fig)
    print(f"saved {path}")


def cmd_tube(args):
    tube = Nanotube(args.n, args.m, gamma0=args.gamma0, overlap=args.overlap)
    print(tube.summary())
    if not args.no_plot:
        print()
        _save(plotting.plot_tube(tube, emax=args.emax), args.out, f"tube_{tube.n}_{tube.m}.png")


def cmd_laser(args):
    e_laser = laser_energy(args.wavelength)
    hits = resonant_tubes(args.wavelength, window=args.window, dmin=args.dmin, dmax=args.dmax,
                          gamma0=args.gamma0, overlap=args.overlap, rbm=args.rbm,
                          rbm_tolerance=args.rbm_tol, rbm_A=args.rbm_A, rbm_B=args.rbm_B)
    print(f"Laser {args.wavelength:g} nm = {e_laser:.3f} eV, "
          f"resonance window +/- {args.window:.2f} eV")
    scope = f"Diameters {args.dmin:.2f}-{args.dmax:.2f} nm"
    if args.rbm is not None:
        scope += f", RBM {args.rbm:g} +/- {args.rbm_tol:g} cm^-1"
    print(scope)
    if not hits:
        print("\nNo resonant tubes found.")
        return
    print(f"\n{'(n,m)':<9}{'type':<6}{'d (nm)':>7}{'RBM (cm^-1)':>13}"
          f"{'E_ii':>9}{'E (eV)':>9}{'detuning':>10}")
    for tube, t in hits:
        rbm = rbm_frequency(tube.diameter, args.rbm_A, args.rbm_B)
        print(f"{tube.name:<9}{'M' if tube.is_metallic else 'S':<6}{tube.diameter:>7.3f}"
              f"{rbm:>13.1f}{t.label:>9}{t.energy:>9.3f}{t.energy - e_laser:>+10.3f}")
    count = f"{len(hits)} resonance" + ("" if len(hits) == 1 else "s")
    print(f"\n{count}. Energies are nearest-neighbour tight-binding estimates "
          f"(gamma0 = {args.gamma0:g} eV);\nconfirm (n,m) assignments against an "
          f"experimental Kataura plot.")


def cmd_figures(args):
    kw = dict(gamma0=args.gamma0, overlap=args.overlap)
    _save(plotting.plot_graphene(**kw), args.out, "graphene.png")
    _save(plotting.plot_cutting_line_comparison([Nanotube(9, 0, **kw), Nanotube(10, 0, **kw)]),
          args.out, "cutting_lines.png")
    for n, m in ((10, 10), (10, 0), (6, 5)):
        _save(plotting.plot_tube(Nanotube(n, m, **kw)), args.out, f"tube_{n}_{m}.png")
    _save(plotting.plot_kataura(**kw), args.out, "kataura.png")


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gamma0", type=float, default=GAMMA0,
                        help="nearest-neighbour hopping energy in eV (default: %(default)s)")
    common.add_argument("--overlap", type=float, default=0.0,
                        help="overlap integral s (default: %(default)s)")

    parser = argparse.ArgumentParser(
        prog="python -m cnt_tb",
        description="Tight-binding band structures of graphene and carbon nanotubes.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tube", parents=[common], help="summary and figure for one (n, m) tube")
    p.add_argument("n", type=int, help="chiral index n")
    p.add_argument("m", type=int, help="chiral index m")
    p.add_argument("--emax", type=float, default=2.5,
                   help="plot energy window in eV (default: %(default)s)")
    p.add_argument("--out", default="figures", help="output folder (default: %(default)s)")
    p.add_argument("--no-plot", action="store_true", help="print the summary only")
    p.set_defaults(func=cmd_tube)

    p = sub.add_parser("laser", parents=[common], help="list tubes resonant with a Raman laser")
    p.add_argument("wavelength", type=float, help="laser wavelength in nm")
    p.add_argument("--window", type=float, default=0.1,
                   help="resonance half-width in eV (default: %(default)s)")
    p.add_argument("--dmin", type=float, default=0.6, help="minimum diameter in nm (default: %(default)s)")
    p.add_argument("--dmax", type=float, default=2.0, help="maximum diameter in nm (default: %(default)s)")
    p.add_argument("--rbm", type=float, help="observed RBM peak in cm^-1 to match")
    p.add_argument("--rbm-tol", type=float, default=5.0,
                   help="RBM matching tolerance in cm^-1 (default: %(default)s)")
    p.add_argument("--rbm-A", type=float, default=RBM_A,
                   help="A in omega = A/d + B (default: %(default)s)")
    p.add_argument("--rbm-B", type=float, default=RBM_B,
                   help="B in omega = A/d + B (default: %(default)s)")
    p.set_defaults(func=cmd_laser)

    p = sub.add_parser("figures", parents=[common], help="regenerate every figure in the README")
    p.add_argument("--out", default="figures", help="output folder (default: %(default)s)")
    p.set_defaults(func=cmd_figures)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

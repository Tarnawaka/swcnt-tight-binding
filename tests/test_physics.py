"""Checks against analytic results and basic physical constraints."""
import numpy as np
import pytest

from cnt_tb import graphene
from cnt_tb.kataura import chiralities, resonant_tubes
from cnt_tb.nanotube import Nanotube
from cnt_tb.raman import diameter_from_rbm, laser_energy, rbm_frequency

G0 = graphene.GAMMA0


# ---------------------------------------------------------------- graphene
def test_reciprocal_lattice():
    A = np.array([graphene.A1, graphene.A2])
    B = np.array([graphene.B1, graphene.B2])
    assert np.allclose(A @ B.T, 2 * np.pi * np.eye(2))


@pytest.mark.parametrize("point, expected", [
    (graphene.GAMMA, 3.0), (graphene.M_POINT, 1.0),
    (graphene.K_POINT, 0.0), (graphene.K_PRIME, 0.0),
])
def test_high_symmetry_points(point, expected):
    """E = +/- gamma0 |f|: 3 gamma0 at Gamma, gamma0 at M, 0 at the Dirac points."""
    w = np.sqrt(graphene.structure_factor_sq(*point))
    assert w == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("overlap", [0.0, 0.129])
def test_closed_form_matches_diagonalisation(overlap):
    rng = np.random.default_rng(0)
    for k in rng.uniform(-2, 2, size=(25, 2)):
        H, S = graphene.hamiltonian(k, overlap=overlap)
        numeric = np.sort(np.linalg.eigvals(np.linalg.solve(S, H)).real)
        ev, ec = graphene.bands(k[0], k[1], overlap=overlap)
        assert np.allclose(numeric, [ev, ec], atol=1e-10)


def test_graphene_dos_normalised():
    E, g = graphene.density_of_states(-9, 9, sigma=0.05, nk=400)
    assert g.sum() * (E[1] - E[0]) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------- geometry
@pytest.mark.parametrize("n, m, atoms", [(10, 10, 40), (10, 0, 40), (6, 5, 364), (5, 5, 20)])
def test_atoms_per_unit_cell(n, m, atoms):
    assert Nanotube(n, m).num_atoms == atoms


@pytest.mark.parametrize("n, m", [(10, 0), (10, 10), (6, 5), (13, 6), (9, 1)])
def test_reciprocal_vectors(n, m):
    t = Nanotube(n, m)
    assert t.Ch @ t.K1 == pytest.approx(2 * np.pi)
    assert t.T @ t.K2 == pytest.approx(2 * np.pi)
    assert t.Ch @ t.K2 == pytest.approx(0, abs=1e-10)
    assert t.T @ t.K1 == pytest.approx(0, abs=1e-10)
    assert np.linalg.norm(t.K2) == pytest.approx(2 * np.pi / t.unit_cell_length)


def test_known_geometry():
    assert Nanotube(10, 10).diameter == pytest.approx(1.356, abs=0.001)
    assert Nanotube(10, 10).chiral_angle == pytest.approx(30.0)
    assert Nanotube(10, 0).chiral_angle == pytest.approx(0.0)
    assert Nanotube(6, 5).chiral_angle == pytest.approx(27.0, abs=0.05)
    assert Nanotube(5, 6).name == "(6,5)"          # mirror image is reordered


# ---------------------------------------------------------------- electronics
def test_metallicity_rule():
    """Numerical gap is zero exactly when (n - m) is divisible by 3."""
    for n in range(1, 13):
        for m in range(n + 1):
            tube = Nanotube(n, m)
            gap = tube.band_gap()
            if tube.is_metallic:
                assert gap < 1e-6, tube
            else:
                assert gap > 0.1, tube


@pytest.mark.parametrize("n", [7, 8, 10, 11, 13, 14, 19])
def test_zigzag_gap_matches_analytic(n):
    """For (n, 0): E_g = 2 gamma0 min_mu |1 + 2 cos(mu pi / n)| (Saito et al.)."""
    mu = np.arange(2 * n)
    expected = 2 * G0 * np.min(np.abs(1 + 2 * np.cos(mu * np.pi / n)))
    assert Nanotube(n, 0).band_gap() == pytest.approx(expected, rel=1e-6)


def test_zigzag_10_0_gap_value():
    assert Nanotube(10, 0).band_gap() == pytest.approx(0.948, abs=0.001)


@pytest.mark.parametrize("n", [5, 8, 10, 12, 15])
def test_armchair_first_transition(n):
    """For (n, n): E11^M = 2 gamma0 sin(pi / n)."""
    first = Nanotube(n, n).transition_energies()[0]
    assert first.label == "E11M"
    assert first.energy == pytest.approx(2 * G0 * np.sin(np.pi / n), rel=1e-6)


def test_gap_scales_as_inverse_diameter():
    """Large semiconducting tubes approach E_g = 2 a_cc gamma0 / d."""
    for n, m in [(17, 0), (13, 6), (14, 10)]:
        tube = Nanotube(n, m)
        estimate = 2 * graphene.A_CC * G0 / (tube.diameter * 10)
        assert tube.band_gap() == pytest.approx(estimate, rel=0.03)


def test_metallic_chiral_tube_has_split_transitions():
    labels = [t.label for t in Nanotube(9, 3).transition_energies()]
    assert labels[:2] == ["E11M-", "E11M+"]


def test_armchair_dos_at_fermi_level():
    """Metallic plateau: sqrt(3) a_cc / (pi^2 gamma0 d) states/eV/atom per spin."""
    tube = Nanotube(10, 10)
    E, g = tube.density_of_states(-0.3, 0.3, sigma=0.01)
    expected = np.sqrt(3) * graphene.A_CC / (np.pi ** 2 * G0 * tube.diameter * 10)
    assert g[np.argmin(np.abs(E))] == pytest.approx(expected, rel=0.02)


def test_semiconductor_dos_empty_in_gap():
    E, g = Nanotube(10, 0).density_of_states(-0.3, 0.3, sigma=0.01)
    assert np.all(g < 1e-8)


@pytest.mark.parametrize("n, m", [(10, 0), (10, 10), (6, 5)])
def test_nanotube_dos_normalised(n, m):
    E, g = Nanotube(n, m).density_of_states(-9, 9, sigma=0.05)
    assert g.sum() * (E[1] - E[0]) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------- spectroscopy
def test_laser_energy():
    assert laser_energy(532) == pytest.approx(2.3305, abs=1e-3)


def test_rbm_round_trip():
    assert diameter_from_rbm(rbm_frequency(1.2)) == pytest.approx(1.2)
    assert diameter_from_rbm(rbm_frequency(1.2, 223.5, 12.5), 223.5, 12.5) == pytest.approx(1.2)


def test_chiralities_in_range_and_sorted():
    tubes = [Nanotube(n, m) for n, m in chiralities(0.8, 1.2)]
    diameters = [t.diameter for t in tubes]
    assert all(0.8 <= d <= 1.2 for d in diameters)
    assert diameters == sorted(diameters)


def test_resonant_tubes_within_window():
    hits = resonant_tubes(532, window=0.1, dmin=0.7, dmax=1.2)
    assert hits
    for tube, transition in hits:
        assert abs(transition.energy - laser_energy(532)) <= 0.1
        assert 0.7 <= tube.diameter <= 1.2

"""Conversions that connect the model to Raman experiments."""
from __future__ import annotations

HC_EV_NM = 1239.84193   # Planck constant times speed of light (eV nm)

# Radial breathing mode (RBM): omega = A / d + B, with d in nm and omega in
# cm^-1. A = 248, B = 0 is the relation for isolated SWCNTs on Si/SiO2 from
# Jorio et al., Phys. Rev. Lett. 86, 1118 (2001). Bundled or
# surfactant-wrapped tubes need different constants, so both are parameters.
RBM_A = 248.0
RBM_B = 0.0


def laser_energy(wavelength_nm):
    """Photon energy (eV) for a wavelength in nm."""
    return HC_EV_NM / wavelength_nm


def photon_wavelength(energy_ev):
    """Photon wavelength (nm) for an energy in eV."""
    return HC_EV_NM / energy_ev


def rbm_frequency(diameter_nm, A=RBM_A, B=RBM_B):
    """RBM frequency (cm^-1) of a tube with the given diameter (nm)."""
    return A / diameter_nm + B


def diameter_from_rbm(omega, A=RBM_A, B=RBM_B):
    """Tube diameter (nm) from a measured RBM frequency (cm^-1)."""
    return A / (omega - B)

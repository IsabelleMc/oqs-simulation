"""Spin-boson model: shared parameters, spectral densities, and bath discretisation.

The spin-boson model is a two-level system (qubit) coupled linearly to a bosonic
bath. The bath's influence on the system is fully characterised by the spectral
density J(omega). Both the chain-mapping and surrogate-Hamiltonian methods use
the same physical model; they differ in how the (infinite) bath is truncated.

References:
    Chin et al., J. Math. Phys. 51, 092109 (2010) — chain mapping
    Leggett et al., Rev. Mod. Phys. 59, 1 (1987) — spin-boson background
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SpinBosonParams:
    """Parameters of the spin-boson model.

    omega0:  system (qubit) frequency
    omega_c: bath cutoff frequency
    alpha:   dimensionless system-bath coupling strength
    s:       Ohmicity (s=1 ohmic, s<1 sub-ohmic, s>1 super-ohmic)
    """

    omega0: float = 1.0
    omega_c: float = 2.0
    alpha: float = 0.001
    s: float = 1.0


def coupling_eta(p: SpinBosonParams) -> float:
    """System-to-first-chain-site coupling for a hard-cutoff spectral density.

    Equivalent to sqrt(integral of J(omega) d omega / pi), per eq. (3) in
    Prior et al., Phys. Rev. Lett. 105, 050404 (2010).
    """
    integral = np.pi * p.alpha * p.omega_c ** 2 / 2.0
    return float(np.sqrt(integral / np.pi))


def discretise_bath_hard_cutoff(L: int, p: SpinBosonParams):
    """Chain frequencies and tunnellings for a hard-cutoff Ohmic bath.

    Returns (epsilon_n, t_n), each of length L, the on-site energies and
    nearest-neighbour tunnellings of the mapped 1D chain. The first chain
    site is reserved for the system, so epsilon_n[0] = t_n[0] = t_n[1] = 0.

    Reference: Chin et al. (2010), eqs. (27)-(28).
    """
    epsilon_n = np.zeros(L)
    t_n = np.zeros(L)
    n = np.arange(L - 1)
    s = p.s
    omega_c = p.omega_c

    epsilon_n[1:] = omega_c / 2 * (1 + s ** 2 / ((s + 2 * n) * (2 + s + 2 * n)))
    nn = n[:-1]
    t_n[2:] = (
        omega_c
        * (1 + nn)
        * (1 + s + nn)
        / ((s + 2 + 2 * nn) * (3 + s + 2 * nn))
        * np.sqrt((3 + s + 2 * nn) / (1 + s + 2 * nn))
    )
    return epsilon_n, t_n


def discretise_bath_exp_cutoff(L: int, p: SpinBosonParams):
    """Chain frequencies and tunnellings for an exponential-cutoff Ohmic bath.

    Reference: Chin et al. (2010), eqs. (32)-(33).
    """
    epsilon_n = np.zeros(L)
    t_n = np.zeros(L)
    n = np.arange(L - 1)
    s = p.s
    omega_c = p.omega_c

    epsilon_n[1:] = omega_c * (2 * n + 1 + s)
    nn = n[:-1]
    t_n[2:] = omega_c * np.sqrt((nn + 1) * (nn + s + 1))
    return epsilon_n, t_n

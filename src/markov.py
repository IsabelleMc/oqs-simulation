"""Markov approximation for the spin-boson model — the standard analytical baseline.

The Markov approximation assumes the bath has no memory: information flows
unidirectionally from system to bath. Valid in the weak-coupling limit only.
We use it as a reference to validate the chain-mapping and surrogate-Hamiltonian
approximations against, in the regime where it should agree with both.
"""

import numpy as np
import qutip as qt

from .spin_boson import SpinBosonParams


def _spectral_density_ohmic(omega: float, p: SpinBosonParams) -> float:
    """Ohmic spectral density with hard cutoff at omega_c."""
    if omega >= p.omega_c:
        return 0.0
    return float(2 * np.pi * p.alpha * p.omega_c ** (1 - p.s) * omega ** p.s)


def evolve_markov(p: SpinBosonParams, t_max: float, rho0: qt.Qobj, n_steps: int = 1000):
    """Evolve the qubit under the Lindblad master equation in the Markovian limit.

    Constructs jump operators from the system eigenbasis and the bath spectral
    density, then propagates rho0 with QuTiP's mesolve. Returns the qutip Result
    with sigma_x, sigma_y, sigma_z expectations.
    """
    H_sys = (p.omega0 / 2) * qt.sigmaz()
    e_vals, e_vecs = H_sys.eigenstates()
    energy_gap = e_vals[1] - e_vals[0]

    g_eff = e_vecs[0].dag() * qt.sigmax() * e_vecs[1]
    gamma = float(np.abs(g_eff) ** 2) * _spectral_density_ohmic(energy_gap, p)
    L_op = np.sqrt(gamma) * (e_vecs[0] * e_vecs[1].dag())

    tlist = np.linspace(0, t_max, n_steps)
    return qt.mesolve(
        H_sys, rho0, tlist, [L_op], [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    )

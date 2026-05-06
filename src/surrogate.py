"""Surrogate Hamiltonian: direct sampling of N bath modes near system resonance.

Truncates the infinite bath to a finite, discrete set of modes sampled
equidistantly around omega_0. The system + truncated bath is then propagated
as a closed system with QuTiP's sesolve.

Reference: Baer & Kosloff, J. Chem. Phys. 106, 8862 (1997).

The geometric constraint on cutoff frequency:
    Modes are sampled in [omega_0 - x, omega_0 + x] with x = omega_c - omega_0.
    To keep all sampled frequencies positive we need omega_c <= 2 * omega_0.
"""

import numpy as np
import qutip as qt

from .spin_boson import SpinBosonParams


def _sample_bath_frequencies(N: int, p: SpinBosonParams):
    """Equidistant sampling around omega_0 with one mode exactly on resonance.

    Returns (X_k, dc) where X_k are the N sampled frequencies and dc is the
    spacing between adjacent modes (used to scale the discretised coupling).
    """
    if not p.omega_c <= 2 * p.omega0:
        raise ValueError(
            f"Surrogate Hamiltonian requires omega_c <= 2*omega_0; "
            f"got omega_c={p.omega_c}, omega_0={p.omega0}"
        )

    min_bath = 2 * p.omega0 - p.omega_c
    lower = np.linspace(min_bath, p.omega0, (N // 2) + 1)
    dc = lower[1] - lower[0] if N > 1 else 1.0

    if N % 2 == 1:
        upper = [p.omega0 + i * dc for i in range(1, N // 2 + 1)]
        X_k = np.concatenate((lower, upper))
    else:
        upper = [p.omega0 + i * dc for i in range(1, N // 2 + 1)]
        X_k = np.concatenate((lower[:-1], upper))
    return X_k, dc


def evolve_surrogate(
    N: int,
    n_levels: int,
    p: SpinBosonParams,
    t_max: float,
    psi_sys0: qt.Qobj,
    n_steps: int = 1000,
):
    """Propagate the spin-boson model with a finite bath of N sampled modes.

    Parameters
    ----------
    N : number of bath modes (odd N includes a mode exactly on resonance)
    n_levels : Fock-space truncation per mode (typically 2-3 in weak coupling)
    p : spin-boson parameters
    t_max : simulation time
    psi_sys0 : initial system state (e.g., qt.basis(2, 0) for spin up)

    Returns
    -------
    result : qutip Result object from sesolve
    expect_dict : dict mapping observable name to expectation-value array
    bath_frequencies : the N sampled bath frequencies
    """
    if p.s != 1.0:
        raise ValueError("Only Ohmic spectral density (s=1) implemented.")

    X_k, _dc = _sample_bath_frequencies(N, p)
    g_k = X_k
    h_k = np.sqrt(_dc * 2 * p.alpha * p.omega_c * X_k)

    sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    I_bosons = [qt.qeye(n_levels) for _ in range(N)]

    # System: omega_0/2 * sigma_z, embedded in full Hilbert space.
    H_sys = qt.tensor((p.omega0 / 2) * sz, *I_bosons)

    # Bath: sum of g_k * a_k^dag a_k for each mode k.
    bath_number = sum(
        qt.tensor(*[
            qt.qeye(n_levels) if j != i else g_k[i] * qt.create(n_levels) * qt.destroy(n_levels)
            for j in range(N)
        ])
        for i in range(N)
    )
    H_bath = qt.tensor(qt.qeye(2), bath_number)

    # Interaction: 0.5 * sigma_x ⊗ sum_k h_k (a_k + a_k^dag).
    bath_x = sum(
        qt.tensor(*[
            qt.qeye(n_levels) if j != i else h_k[i] * (qt.destroy(n_levels) + qt.create(n_levels))
            for j in range(N)
        ])
        for i in range(N)
    )
    H_int = qt.tensor(0.5 * sx, bath_x)

    H = H_sys + H_bath + H_int
    psi0 = qt.tensor(psi_sys0, *[qt.basis(n_levels, 0) for _ in range(N)])
    tlist = np.linspace(0, t_max, n_steps)

    bath_total_number = qt.tensor(qt.qeye(2), bath_number / max(g_k))  # tracking only
    e_ops = {
        "sx": qt.tensor(sx, *I_bosons),
        "sy": qt.tensor(sy, *I_bosons),
        "sz": qt.tensor(sz, *I_bosons),
        "bath_n": bath_total_number,
    }

    result = qt.sesolve(H, psi0, tlist, e_ops=list(e_ops.values()), progress_bar=True)
    expect_dict = {name: result.expect[i] for i, name in enumerate(e_ops.keys())}
    return result, expect_dict, X_k

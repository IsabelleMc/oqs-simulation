"""Chain mapping: unitary transform of the bath into a 1D nearest-neighbour chain.

The continuous bath is mapped onto a semi-infinite 1D chain via orthogonal
polynomials (Jacobi for hard cutoff, Laguerre for exponential cutoff). The
chain is truncated to L sites and propagated alongside the system as an MPS
with TEBD (via TeNPy).

The first chain site holds the qubit; subsequent sites are bath modes ordered
so that the most strongly correlated modes sit nearest the system.

Reference: Chin et al., J. Math. Phys. 51, 092109 (2010).
"""

import time
import numpy as np

import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import tebd
from tenpy.models import CouplingMPOModel, MPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.site import Site, kron

from .spin_boson import SpinBosonParams, coupling_eta, discretise_bath_hard_cutoff


class CMSite(Site):
    """Site that holds either a qubit (first site) or a truncated bosonic mode.

    The bosonic Fock space is truncated at dimension d. The qubit operators
    (sx, sy, sz, ...) are embedded into the d-dimensional space so that all
    sites have the same local dimension.
    """

    def __init__(self, d=3, sort_charge=True):
        self.d = d

        # Qubit operators (2x2), embedded into d-dimensional space.
        sp = np.array([[0, 1], [0, 0]])
        sm = sp.T
        sx = sp + sm
        sy = (sm - sp) * 1j
        sz = np.array([[1, 0], [0, -1]])
        s0 = np.identity(2)

        sp = self._embed(sp)
        sm = self._embed(sm)
        sx = self._embed(sx)
        sy = self._embed(sy)
        sz = self._embed(sz)
        s0 = self._embed(s0)

        # Bosonic operators (d-dimensional truncation).
        bp = np.diag(np.sqrt(np.arange(1, self.d)), k=-1)
        bm = bp.T
        bn = np.diag(np.arange(self.d))
        bx = bp + bm
        by = (bm - bp) * 1j
        b0 = np.diag(np.ones(self.d))
        bmax = np.zeros((self.d, self.d))
        bmax[self.d - 1, self.d - 1] = 1

        ops = dict(sp=sp, sm=sm, sx=sx, sy=sy, sz=sz, s0=s0,
                   bp=bp, bm=bm, bn=bn, bx=bx, by=by, b0=b0, bmax=bmax)
        leg = npc.LegCharge.from_trivial(self.d)
        self.conserve = None
        Site.__init__(self, leg, sort_charge=sort_charge, **ops)

    def __repr__(self):
        return f"CMSite(d={self.d})"

    def _embed(self, op):
        """Embed a 2x2 qubit operator into the d-dimensional site Hilbert space."""
        if op.shape[0] > self.d:
            raise ValueError(f"Operator too large to embed: {op.shape} > d={self.d}")
        if op.shape[0] == self.d:
            return op
        out = np.zeros((self.d, self.d), dtype=complex)
        out[: op.shape[0], : op.shape[0]] = op
        return out


class CMModel(NearestNeighborModel, CouplingMPOModel):
    """1D chain Hamiltonian: qubit on site 0, bath modes on sites 1..L-1.

    Required model_params:
        L         : chain length (qubit + L-1 bath modes)
        d         : per-site Fock truncation
        omega0    : qubit frequency
        eta       : qubit-to-first-bath-site coupling
        epsilon_n : array of length L, bath site energies (epsilon_n[0] unused)
        t_n       : array of length L, nearest-neighbour tunnellings
                    (t_n[0], t_n[1] unused; t_n[i] = chain link i-1 <-> i)
    """

    explicit_plus_hc = False

    def __init__(self, model_params):
        L = model_params.get("L", 2)
        self.L = L
        sort_charge = model_params.get("sort_charge", True)
        d = model_params.get("d", 3)
        site = CMSite(d, sort_charge=sort_charge)
        omega0 = model_params["omega0"]
        eta = model_params["eta"]
        t_n = model_params["t_n"]
        epsilon_n = model_params["epsilon_n"]

        lat = Chain(self.L, site, bc="open", bc_MPS="finite")
        self.lat = lat

        sz, sx = site.sz, site.sx
        bx, bp, bm, bn, b0 = site.bx, site.bp, site.bm, site.bn, site.b0

        # Site interaction operators, transposed for tenpy bond conventions.
        h_int = kron(sx, bx).split_legs().transpose(["p0", "p1", "p0*", "p1*"])
        h_chain = (kron(bp, bm) + kron(bm, bp)).split_legs().transpose(
            ["p0", "p1", "p0*", "p1*"]
        )

        # Bond Hamiltonians: H_bond[i] acts on sites (i-1, i).
        H_bond = [t * h_chain for t in t_n]
        H_bond[1] = eta * h_int
        H_bond[1] += (omega0 / 2) * (
            kron(sz, b0).split_legs().transpose(["p0", "p1", "p0*", "p1*"])
        )
        for i, eps in enumerate(epsilon_n):
            H_bond[i] += eps * (
                kron(b0, bn).split_legs().transpose(["p0", "p1", "p0*", "p1*"])
            )
        H_bond[0] = None

        NearestNeighborModel.__init__(self, lat, H_bond)
        MPOModel.__init__(self, lat, self.calc_H_MPO_from_bond())


def _measurement(eng, data, L):
    """Append observables at the current evolved time to the data dict.

    Tracks system expectations (sx, sy, sz, sp), local bath occupation (bn),
    boundary occupation (bmax — diverges from zero only when finite-size
    effects start to corrupt the dynamics), entanglement entropy, and
    truncation error.
    """
    keys = ["t", "entropy", "sz", "sx", "sy", "sp", "bn", "bmax",
            "trunc_err", "wall_time", "sz_bn", "sp_bn"]
    if data is None:
        data = {k: [] for k in keys}
    data["t"].append(eng.evolved_time)
    data["entropy"].append(eng.psi.entanglement_entropy())
    data["sx"].append(eng.psi.expectation_value("sx"))
    data["sy"].append(eng.psi.expectation_value("sy"))
    data["sz"].append(eng.psi.expectation_value("sz"))
    data["sp"].append(eng.psi.expectation_value("sp"))
    data["bn"].append(eng.psi.expectation_value("bn"))
    data["bmax"].append(eng.psi.expectation_value("bmax"))
    data["trunc_err"].append(eng.trunc_err.eps)
    data["wall_time"].append(time.time())
    data["sz_bn"].append(
        eng.psi.correlation_function("sz", "bn", sites1=[0], sites2=np.arange(1, L))
    )
    data["sp_bn"].append(
        eng.psi.correlation_function("sp", "bn", sites1=[0], sites2=np.arange(1, L))
    )
    return data


def evolve_chain_mapping(
    L: int,
    p: SpinBosonParams,
    t_max: float,
    d: int = 3,
    chi_max: int = 50,
    dt: float = 0.1,
    log_level: str = "WARNING",
):
    """Run TEBD on the chain-mapped spin-boson model.

    Parameters
    ----------
    L : chain length (qubit + L-1 bath modes)
    p : spin-boson parameters
    t_max : simulation time
    d : per-site Fock truncation
    chi_max : MPS bond dimension cap
    dt : Trotter step
    log_level : tenpy log level (DEBUG, INFO, WARNING, ERROR)

    Returns
    -------
    data : dict of np.ndarrays with observables vs time
    model_params : dict of resolved parameters used in the run
    """
    eta = coupling_eta(p)
    epsilon_n, t_n = discretise_bath_hard_cutoff(L, p)

    model_params = {
        "L": L,
        "d": d,
        "omega0": p.omega0,
        "eta": eta,
        "epsilon_n": epsilon_n,
        "t_n": t_n,
    }

    site = CMSite(d)
    model = CMModel(model_params)

    p_state = [0] * L
    p_state[0] = np.array([1, 0, 0])  # qubit in |up>
    psi = MPS.from_product_state([site] * L, p_state, bc="finite", dtype=complex)

    tebd_params = {
        "N_steps": 1,
        "dt": dt,
        "order": 4,
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-12},
    }
    eng = tebd.TEBDEngine(psi, model, tebd_params)
    tenpy.tools.misc.setup_logging(to_stdout=log_level)

    data = _measurement(eng, None, L)
    while eng.evolved_time < t_max:
        eng.run()
        _measurement(eng, data, L)

    for key in data:
        data[key] = np.array(data[key])
    return data, model_params

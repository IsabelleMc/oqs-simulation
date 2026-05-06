# Simulating Open Quantum Systems with Memory

Code from my MSc Physics thesis at FU Berlin (2025), supervised by
Prof. Christiane Koch.

## The problem in one paragraph

A quantum system in contact with its environment loses information to that
environment — this is why qubits decohere. Standard simulation methods assume
the environment forgets immediately (the *Markov approximation*). Real
environments don't: they retain memory and can feed information back into the
system. Capturing that memory is computationally hard because it requires
tracking the environment's state alongside the system's, and the environment
has, in principle, infinitely many degrees of freedom.

## What this code does

Implements and compares two methods that make the problem tractable by
representing the environment with a *finite, structured* set of modes:

1. **Chain mapping.** A unitary transform (Chin et al., 2010) rewrites the
   environment as a 1D chain with only nearest-neighbour interactions. The
   chain is propagated as a matrix product state with TEBD, via
   [TeNPy](https://tenpy.readthedocs.io/). Cost scales linearly in the
   number of retained modes.

2. **Surrogate Hamiltonian.** Directly sample N environment modes near
   resonance with the system and propagate the joint state with standard
   ODE solvers, via [QuTiP](https://qutip.org/). Simpler to set up, but
   sensitive to how the modes are sampled and constrained to
   `omega_c <= 2 * omega_0`.

Both are tested on the spin-boson model (one qubit + bosonic bath) and
benchmarked against the **Markov approximation**, the standard analytical
baseline that's valid in the weak-coupling limit.

## What the thesis found

- Both methods agree with the Markov approximation in the weak-coupling
  limit, validating the implementations.
- Chain mapping captures non-Markovian features (oscillations, finite
  steady states, information backflow) at strong coupling, across a wide
  range of bath cutoff frequencies.
- The surrogate Hamiltonian is efficient at short times in weak coupling
  but its `omega_c <= 2 * omega_0` constraint limits its applicability to
  broad-spectrum environments.
- Convergence time scales linearly with bath size in both methods,
  governed by the velocity at which information propagates into the
  truncated bath before reflecting at the boundary
  (a Lieb-Robinson-type bound).

## Why this generalises beyond physics

The core idea is: a high-dimensional state with long-range correlations is
reformulated as a 1D structure where correlations are local. Once
correlations are local, computation becomes cheap. The same principle
underlies tensor networks, hidden Markov models, and (loosely) state-space
sequence models in ML.

## Repo layout

```
src/
  spin_boson.py      # shared parameters, spectral densities, bath discretisation
  markov.py          # Markov approximation (analytical baseline)
  surrogate.py       # surrogate Hamiltonian (QuTiP)
  chain_mapping.py   # chain mapping + TEBD (TeNPy)
notebooks/
  demo.ipynb         # weak-coupling validation (Fig. 4.2) and strong-coupling
                     # comparison (Fig. 4.4) where Markov breaks down
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/demo.ipynb
```

The demo notebook produces both the weak-coupling validation plot (where
all three methods agree, confirming the implementations are correct) and
the strong-coupling plot (where the Markov approximation predicts full
relaxation but the other two methods show a finite steady state — the
non-Markovian signature). End to end runs in under a minute on a laptop.

## References

- Chin et al., *J. Math. Phys.* 51, 092109 (2010) — chain mapping
- Baer & Kosloff, *J. Chem. Phys.* 106, 8862 (1997) — surrogate Hamiltonian
- Prior et al., *Phys. Rev. Lett.* 105, 050404 (2010) — TEBD on chain-mapped baths
- de Vega & Alonso, *Rev. Mod. Phys.* 89, 015001 (2017) — non-Markovian review
- Leggett et al., *Rev. Mod. Phys.* 59, 1 (1987) — spin-boson background

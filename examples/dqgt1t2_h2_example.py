"""Real H2 v2RDM/HF example for constructing T1/T2 endpoint matrices.

Run from an installed Psi4 + Hilbert environment, for example from the repository
root after building/installing the plugin::

    python examples/dqgt1t2_h2_example.py

The example mirrors the small v2RDM workflows in ``tests/v2rdm*/input.dat`` but
keeps the active space tiny (H2/STO-3G, two active spatial orbitals) so the dense
T1/T2 matrices are only 8 x 8.
"""

from __future__ import annotations

import numpy as np
import psi4

import hilbert
from lambda_dqgt import construct_t1_t2_endpoint_pair, optimize_lambda_for_dqgt1t2


def _psi4_matrix_to_array(matrix) -> np.ndarray:
    """Convert a Psi4 Matrix with one C1 block to a NumPy array."""

    return np.asarray(matrix, dtype=float)


def _tpdm_matrix_to_tensor(tpdm_matrix: np.ndarray, n_orbitals: int) -> np.ndarray:
    """Convert Hilbert's flattened TPDM matrix to Gamma(p, q; r, s)."""

    return np.asarray(tpdm_matrix, dtype=float).reshape(n_orbitals, n_orbitals, n_orbitals, n_orbitals)


def main() -> None:
    h2 = psi4.geometry(
        """
        0 1
        H
        H 1 0.74
        symmetry c1
        """
    )

    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "pk",
            "e_convergence": 1.0e-10,
            "d_convergence": 1.0e-10,
            "restricted_docc": [0],
            "active": [2],
        }
    )
    psi4.set_module_options(
        "hilbert",
        {
            "positivity": "dqg",
            "maxiter": 20000,
            "r_convergence": 1.0e-6,
            "e_convergence": 1.0e-8,
        },
    )

    scf_energy, ref_wfn = psi4.energy("scf", molecule=h2, return_wfn=True)
    options = psi4.core.get_options()
    options.set_current_module("HILBERT")

    v2rdm = hilbert.v2RDMHelper(ref_wfn, options)
    v2rdm_energy = v2rdm.compute_energy()

    v2rdm_opdm = _psi4_matrix_to_array(v2rdm.get_opdm())
    n_active = v2rdm_opdm.shape[0]
    v2rdm_tpdm = _tpdm_matrix_to_tensor(_psi4_matrix_to_array(v2rdm.get_tpdm()), n_active)

    # Closed-shell H2/STO-3G has two electrons in the lowest active spatial MO
    # at the Hartree-Fock endpoint.  The T1/T2 builders consume the same active
    # spatial-orbital ordering used by the Hilbert v2RDM helper.
    hf_opdm = np.zeros_like(v2rdm_opdm)
    hf_opdm[0, 0] = 2.0

    (t1_v2rdm, t1_hf), (t2_v2rdm, t2_hf) = construct_t1_t2_endpoint_pair(
        v2rdm_tpdm,
        hf_opdm,
        v2rdm_opdm=v2rdm_opdm,
    )

    result = optimize_lambda_for_dqgt1t2(
        dqg_energy=v2rdm_energy,
        hf_energy=scf_energy,
        t1_block_pairs=[(t1_v2rdm, t1_hf)],
        t2_block_pairs=[(t2_v2rdm, t2_hf)],
    )

    print(f"SCF energy:       {scf_energy: .12f}")
    print(f"v2RDM energy:     {v2rdm_energy: .12f}")
    print(f"T1 block shape:   {t1_v2rdm.shape}")
    print(f"T2 block shape:   {t2_v2rdm.shape}")
    print(f"lambda:           {result.lambda_value: .12f}")
    print(f"lambda energy:    {result.energy: .12f}")
    print(f"minimum eigenval: {result.minimum_eigenvalue: .12e}")


if __name__ == "__main__":
    main()

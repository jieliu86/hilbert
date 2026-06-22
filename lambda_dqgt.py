#
# @BEGIN LICENSE
#
# Hilbert: a space for quantum chemistry plugins to Psi4
#
# Copyright (c) 2020 by its authors (LICENSE).
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.
#
# @END LICENSE
#

"""One-dimensional HF/DQG interpolation utilities for DQGT1T2 checks.

This module implements the final scalar optimization step for the workflow
where a v2RDM-DQG 2-RDM and a Hartree-Fock 2-RDM are linearly interpolated,

    D2(lambda) = (1 - lambda) D2_DQG + lambda D2_HF,

and the DQGT1T2 feasibility of the interpolated point is tested by checking
positive semidefiniteness of T1/T2 endpoint matrices supplied by the caller.
This module does not construct T1/T2 matrices from v2RDM-CASSCF internals: the
``t1.cc`` and ``t2.cc`` files implement SDP constraint operators, so endpoint
T1/T2 matrices should be supplied only after they have been constructed by a
validated v2RDM-CASSCF-side map or another trusted implementation.  The endpoint
matrices must already be expressed in the same orbital basis and with the same
spin/symmetry block layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[Sequence[float]]
BlockPair = tuple[ArrayLike, ArrayLike]


@dataclass(frozen=True)
class LambdaDQGTResult:
    """Result of the one-dimensional HF/DQG interpolation search."""

    lambda_value: float
    energy: float
    minimum_eigenvalue: float
    feasible: bool
    iterations: int


def _as_symmetric_matrix(matrix: ArrayLike, label: str) -> np.ndarray:
    """Return a dense symmetric matrix after basic validation."""

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{label} must be a square matrix; got shape {array.shape}.")
    if not np.allclose(array, array.T, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError(f"{label} must be symmetric/Hermitian within tolerance.")
    return 0.5 * (array + array.T)


def _normalize_block_pairs(block_pairs: Iterable[BlockPair]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validate endpoint block pairs and convert them to dense arrays."""

    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    for block_id, (dqg_block, hf_block) in enumerate(block_pairs):
        dqg = _as_symmetric_matrix(dqg_block, f"DQG block {block_id}")
        hf = _as_symmetric_matrix(hf_block, f"HF block {block_id}")
        if dqg.shape != hf.shape:
            raise ValueError(
                f"DQG block {block_id} shape {dqg.shape} does not match "
                f"HF block {block_id} shape {hf.shape}."
            )
        normalized.append((dqg, hf))
    if not normalized:
        raise ValueError("At least one T1/T2 block pair is required.")
    return normalized


def interpolated_block(dqg_block: ArrayLike, hf_block: ArrayLike, lambda_value: float) -> np.ndarray:
    """Return ``(1 - lambda) * dqg_block + lambda * hf_block``."""

    dqg = _as_symmetric_matrix(dqg_block, "DQG block")
    hf = _as_symmetric_matrix(hf_block, "HF block")
    if dqg.shape != hf.shape:
        raise ValueError(f"DQG block shape {dqg.shape} does not match HF block shape {hf.shape}.")
    return (1.0 - lambda_value) * dqg + lambda_value * hf


def minimum_interpolated_eigenvalue(
    block_pairs: Iterable[BlockPair],
    lambda_value: float,
) -> float:
    """Return the smallest eigenvalue over all interpolated block pairs.

    The eigenvalues are computed after forming each interpolated matrix.  This
    intentionally does not interpolate endpoint eigenvalues, which would only be
    valid for simultaneously diagonalizable endpoint matrices.
    """

    if lambda_value < 0.0 or lambda_value > 1.0:
        raise ValueError("lambda_value must be in [0, 1].")

    blocks = _normalize_block_pairs(block_pairs)
    min_eval = np.inf
    for dqg, hf in blocks:
        matrix = (1.0 - lambda_value) * dqg + lambda_value * hf
        eval_min = float(np.linalg.eigvalsh(matrix)[0])
        min_eval = min(min_eval, eval_min)
    return float(min_eval)


def is_feasible(
    block_pairs: Iterable[BlockPair],
    lambda_value: float,
    tolerance: float = 1.0e-10,
) -> bool:
    """Return whether all interpolated blocks are positive semidefinite."""

    return minimum_interpolated_eigenvalue(block_pairs, lambda_value) >= -tolerance


def optimize_lambda_for_dqgt1t2(
    dqg_energy: float,
    hf_energy: float,
    t1_block_pairs: Iterable[BlockPair],
    t2_block_pairs: Iterable[BlockPair],
    tolerance: float = 1.0e-10,
    max_iterations: int = 100,
) -> LambdaDQGTResult:
    """Optimize the HF/DQG interpolation parameter under T1/T2 PSD checks.

    Parameters
    ----------
    dqg_energy
        Energy evaluated with the DQG endpoint 2-RDM.
    hf_energy
        Energy evaluated with the HF endpoint 2-RDM.
    t1_block_pairs, t2_block_pairs
        Iterables of ``(DQG block, HF block)`` pairs for T1 and T2.  The DQG
        blocks should be constructed by applying the same T1/T2 maps to the DQG
        2-RDM; they are not read from the DQG SDP because DQG has no T1/T2
        primal blocks.
    tolerance
        Eigenvalue feasibility tolerance.
    max_iterations
        Maximum bisection steps used when the DQG endpoint is infeasible and the
        HF endpoint is feasible.

    Returns
    -------
    LambdaDQGTResult
        The selected lambda, interpolated energy, limiting eigenvalue, and
        feasibility status.

    Notes
    -----
    If ``dqg_energy <= hf_energy``, the minimum-energy feasible point is the
    smallest feasible lambda.  If the HF endpoint is lower in energy, the optimum
    over the interval is lambda = 1 provided that endpoint is feasible.
    """

    t1_blocks = _normalize_block_pairs(t1_block_pairs)
    t2_blocks = _normalize_block_pairs(t2_block_pairs)
    all_blocks = [*t1_blocks, *t2_blocks]

    def min_eval(lambda_value: float) -> float:
        return minimum_interpolated_eigenvalue(all_blocks, lambda_value)

    def feasible(lambda_value: float) -> bool:
        return min_eval(lambda_value) >= -tolerance

    if hf_energy < dqg_energy:
        lambda_value = 1.0
        endpoint_min_eval = min_eval(lambda_value)
        return LambdaDQGTResult(
            lambda_value=lambda_value,
            energy=hf_energy,
            minimum_eigenvalue=endpoint_min_eval,
            feasible=endpoint_min_eval >= -tolerance,
            iterations=0,
        )

    if feasible(0.0):
        endpoint_min_eval = min_eval(0.0)
        return LambdaDQGTResult(
            lambda_value=0.0,
            energy=dqg_energy,
            minimum_eigenvalue=endpoint_min_eval,
            feasible=True,
            iterations=0,
        )

    hf_min_eval = min_eval(1.0)
    if hf_min_eval < -tolerance:
        raise ValueError(
            "The HF endpoint is not feasible for the supplied T1/T2 blocks; "
            "cannot bracket a feasible lambda in [0, 1]."
        )

    low = 0.0
    high = 1.0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        mid = 0.5 * (low + high)
        if feasible(mid):
            high = mid
        else:
            low = mid
        if high - low <= tolerance:
            break

    lambda_value = high
    energy = (1.0 - lambda_value) * dqg_energy + lambda_value * hf_energy
    endpoint_min_eval = min_eval(lambda_value)
    return LambdaDQGTResult(
        lambda_value=lambda_value,
        energy=energy,
        minimum_eigenvalue=endpoint_min_eval,
        feasible=endpoint_min_eval >= -tolerance,
        iterations=iterations,
    )

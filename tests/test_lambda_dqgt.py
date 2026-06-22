import numpy as np

from lambda_dqgt import (
    interpolated_block,
    minimum_interpolated_eigenvalue,
    optimize_lambda_for_dqgt1t2,
)


def test_interpolated_block_forms_matrix_before_diagonalization():
    dqg = np.array([[0.0, 0.3], [0.3, 1.0]])
    hf = np.eye(2)

    block = interpolated_block(dqg, hf, 0.25)

    expected = np.array([[0.25, 0.225], [0.225, 1.0]])
    assert np.allclose(block, expected)
    assert np.isclose(minimum_interpolated_eigenvalue([(dqg, hf)], 0.25), np.linalg.eigvalsh(expected)[0])


def test_optimizes_smallest_feasible_lambda_when_dqg_energy_is_lower():
    dqg_t1 = np.diag([-1.0, 1.0])
    hf_t1 = np.eye(2)
    dqg_t2 = np.diag([0.2, 0.4])
    hf_t2 = np.eye(2)

    result = optimize_lambda_for_dqgt1t2(
        dqg_energy=-2.0,
        hf_energy=-1.0,
        t1_block_pairs=[(dqg_t1, hf_t1)],
        t2_block_pairs=[(dqg_t2, hf_t2)],
        tolerance=1.0e-12,
    )

    assert result.feasible
    assert np.isclose(result.lambda_value, 0.5, atol=1.0e-10)
    assert np.isclose(result.energy, -1.5, atol=1.0e-10)
    assert result.minimum_eigenvalue >= -1.0e-10


def test_selects_hf_endpoint_when_hf_energy_is_lower():
    dqg_t1 = np.diag([-1.0, 1.0])
    hf_t1 = np.eye(2)
    dqg_t2 = np.diag([0.2, 0.4])
    hf_t2 = np.eye(2)

    result = optimize_lambda_for_dqgt1t2(
        dqg_energy=-1.0,
        hf_energy=-2.0,
        t1_block_pairs=[(dqg_t1, hf_t1)],
        t2_block_pairs=[(dqg_t2, hf_t2)],
    )

    assert result.feasible
    assert result.lambda_value == 1.0
    assert result.energy == -2.0

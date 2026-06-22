import numpy as np

from lambda_dqgt import (
    construct_t1_matrix,
    construct_t1_t2_endpoint_pair,
    construct_t2_matrix,
    hartree_fock_two_rdm,
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



def test_constructs_t1_t2_endpoint_blocks_from_v2rdm_and_hf_rdms():
    hf_opdm = np.diag([1.0, 1.0, 0.0])
    v2rdm_tpdm = hartree_fock_two_rdm(hf_opdm)

    (t1_v2rdm, t1_hf), (t2_v2rdm, t2_hf) = construct_t1_t2_endpoint_pair(
        v2rdm_tpdm,
        hf_opdm,
        electron_count=2,
    )

    assert t1_v2rdm.shape == (27, 27)
    assert t2_v2rdm.shape == (27, 27)
    assert np.allclose(t1_v2rdm, t1_hf)
    assert np.allclose(t2_v2rdm, t2_hf)
    assert np.allclose(t1_v2rdm, t1_v2rdm.T)
    assert np.allclose(t2_v2rdm, t2_v2rdm.T)


def test_t1_t2_builders_can_contract_opdm_from_tpdm():
    hf_opdm = np.diag([1.0, 1.0, 0.0])
    tpdm = hartree_fock_two_rdm(hf_opdm)

    assert np.allclose(construct_t1_matrix(tpdm, electron_count=2), construct_t1_matrix(tpdm, hf_opdm))
    assert np.allclose(construct_t2_matrix(tpdm, electron_count=2), construct_t2_matrix(tpdm, hf_opdm))

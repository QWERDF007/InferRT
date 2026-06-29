"""DBSCAN/HDBSCAN ops Python binding tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _load_cluster_data(repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    data_path = repo_root / "assets" / "dbscan_testdata.txt"
    lines = data_path.read_text(encoding="utf-8").strip().splitlines()
    rows = [[float(value) for value in line.split(",")] for line in lines[1:]]
    samples = np.asarray([row[:3] for row in rows], dtype=np.float32)
    labels = np.asarray([int(row[3]) for row in rows], dtype=np.int64)
    assert samples.shape[0] == int(lines[0])
    return samples, labels


def _assert_same_partition(actual: list[int], expected: np.ndarray) -> None:
    actual_array = np.asarray(actual, dtype=np.int64)
    assert actual_array.shape == expected.shape
    np.testing.assert_array_equal(
        actual_array[:, None] == actual_array[None, :],
        expected[:, None] == expected[None, :],
    )


def _algorithm_cases(ops_module):
    return [
        (ops_module.ClusteringAlgorithm.Auto, "auto"),
        (ops_module.ClusteringAlgorithm.Brute, "brute"),
        (ops_module.ClusteringAlgorithm.KDTree, "kd_tree"),
        (ops_module.ClusteringAlgorithm.BallTree, "ball_tree"),
    ]


@pytest.mark.parametrize("algorithm_index", range(4))
def test_dbscan_matches_sklearn_on_asset_data(ops_module, repo_root: Path, algorithm_index: int) -> None:
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    samples, target_labels = _load_cluster_data(repo_root)
    algorithm, sklearn_algorithm = _algorithm_cases(ops_module)[algorithm_index]
    config = ops_module.DBSCANConfig()
    config.eps = 0.8
    config.min_samples = 4
    config.algorithm = algorithm
    config.leaf_size = 8

    result = ops_module.dbscan(samples, config)
    expected = sklearn_cluster.DBSCAN(
        eps=config.eps,
        min_samples=config.min_samples,
        algorithm=sklearn_algorithm,
        leaf_size=config.leaf_size,
    ).fit_predict(samples)

    np.testing.assert_array_equal(np.asarray(result.labels, dtype=np.int64), expected)
    assert len(set(result.labels)) == 7
    assert -1 not in result.labels
    _assert_same_partition(result.labels, target_labels)


@pytest.mark.parametrize("algorithm_index", range(4))
def test_hdbscan_matches_sklearn_on_asset_data(ops_module, repo_root: Path, algorithm_index: int) -> None:
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    if not hasattr(sklearn_cluster, "HDBSCAN"):
        pytest.skip("sklearn.cluster.HDBSCAN is not available")

    samples, target_labels = _load_cluster_data(repo_root)
    algorithm, sklearn_algorithm = _algorithm_cases(ops_module)[algorithm_index]
    config = ops_module.HDBSCANConfig()
    config.min_cluster_size = 5
    config.min_samples = 5
    config.algorithm = algorithm
    config.leaf_size = 8

    result = ops_module.hdbscan(samples, config)
    reference = sklearn_cluster.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_epsilon=config.cluster_selection_epsilon,
        max_cluster_size=None if config.max_cluster_size == 0 else config.max_cluster_size,
        alpha=config.alpha,
        algorithm=sklearn_algorithm,
        leaf_size=config.leaf_size,
        cluster_selection_method="eom",
        allow_single_cluster=config.allow_single_cluster,
        copy=True,
    ).fit(samples)

    np.testing.assert_array_equal(np.asarray(result.labels, dtype=np.int64), reference.labels_)
    np.testing.assert_allclose(np.asarray(result.probabilities, dtype=np.float64), reference.probabilities_)
    assert len(set(result.labels)) == 7
    assert -1 not in result.labels
    assert len(result.probabilities) == len(result.labels)
    assert all(0.0 <= probability <= 1.0 for probability in result.probabilities)
    _assert_same_partition(result.labels, target_labels)


def test_clustering_rejects_invalid_sample_shape(ops_module) -> None:
    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.dbscan(np.zeros((2, 3, 1), dtype=np.float32), ops_module.DBSCANConfig())

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.hdbscan(np.zeros((2, 3, 1), dtype=np.float32), ops_module.HDBSCANConfig())

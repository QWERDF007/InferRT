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


def _assert_same_partition(actual: list[int] | np.ndarray, expected: np.ndarray) -> None:
    actual_array = np.asarray(actual, dtype=np.int64)
    assert actual_array.shape == expected.shape
    np.testing.assert_array_equal(
        actual_array[:, None] == actual_array[None, :],
        expected[:, None] == expected[None, :],
    )


def _count_clusters(labels: list[int] | np.ndarray) -> int:
    return len({int(label) for label in labels if int(label) != -1})


def _algorithm_case(ops_module, name: str):
    return {
        "auto": (ops_module.ClusteringAlgorithm.Auto, "auto"),
        "brute": (ops_module.ClusteringAlgorithm.Brute, "brute"),
        "kd_tree": (ops_module.ClusteringAlgorithm.KDTree, "kd_tree"),
        "ball_tree": (ops_module.ClusteringAlgorithm.BallTree, "ball_tree"),
    }[name]


def _cluster_selection_method(ops_module, name: str):
    return {
        "eom": (ops_module.HDBSCANClusterSelectionMethod.Eom, "eom"),
        "leaf": (ops_module.HDBSCANClusterSelectionMethod.Leaf, "leaf"),
    }[name]


def _metric_case(ops_module, name: str):
    if name == "default":
        return None, "cosine"
    return {
        "euclidean": (ops_module.ClusteringMetric.Euclidean, "euclidean"),
        "cosine": (ops_module.ClusteringMetric.Cosine, "cosine"),
        "manhattan": (ops_module.ClusteringMetric.Manhattan, "manhattan"),
        "minkowski": (ops_module.ClusteringMetric.Minkowski, "minkowski"),
    }[name]


def _sklearn_algorithm_for_metric(sklearn_algorithm: str, sklearn_metric: str) -> str:
    return "brute" if sklearn_metric != "euclidean" else sklearn_algorithm


DBSCAN_PARAMETER_CASES = [
    pytest.param(
        {
            "eps": 0.8,
            "min_samples": 4,
            "algorithm": "auto",
            "leaf_size": 8,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="auto_eps0.8_min4_leaf8",
    ),
    pytest.param(
        {
            "eps": 0.8,
            "min_samples": 4,
            "algorithm": "brute",
            "leaf_size": 8,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="brute_eps0.8_min4_leaf8",
    ),
    pytest.param(
        {
            "eps": 0.8,
            "min_samples": 4,
            "algorithm": "kd_tree",
            "leaf_size": 8,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="kd_tree_eps0.8_min4_leaf8",
    ),
    pytest.param(
        {
            "eps": 0.8,
            "min_samples": 4,
            "algorithm": "ball_tree",
            "leaf_size": 8,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="ball_tree_eps0.8_min4_leaf8",
    ),
    pytest.param(
        {
            "eps": 0.55,
            "min_samples": 2,
            "algorithm": "brute",
            "leaf_size": 4,
            "clusters": 13,
            "noise": 10,
            "target": False,
        },
        id="brute_eps0.55_min2_leaf4",
    ),
    pytest.param(
        {
            "eps": 0.65,
            "min_samples": 4,
            "algorithm": "kd_tree",
            "leaf_size": 16,
            "clusters": 7,
            "noise": 5,
            "target": False,
        },
        id="kd_tree_eps0.65_min4_leaf16",
    ),
    pytest.param(
        {
            "eps": 1.05,
            "min_samples": 6,
            "algorithm": "ball_tree",
            "leaf_size": 12,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="ball_tree_eps1.05_min6_leaf12",
    ),
    pytest.param(
        {
            "eps": 0.45,
            "min_samples": 4,
            "algorithm": "auto",
            "leaf_size": 8,
            "clusters": 14,
            "noise": 81,
            "target": False,
        },
        id="auto_eps0.45_min4_leaf8",
    ),
    pytest.param(
        {
            "eps": 0.01,
            "min_samples": 4,
            "algorithm": "auto",
            "leaf_size": 8,
            "metric": "default",
            "clusters": 7,
            "noise": 32,
            "target": False,
        },
        id="default_cosine_eps0.01_min4_leaf8",
    ),
    pytest.param(
        {
            "eps": 1.0,
            "min_samples": 4,
            "algorithm": "brute",
            "leaf_size": 8,
            "metric": "manhattan",
            "clusters": 7,
            "noise": 1,
            "target": False,
        },
        id="manhattan_eps1.0_min4_leaf8",
    ),
    pytest.param(
        {
            "eps": 1.0,
            "min_samples": 4,
            "algorithm": "auto",
            "leaf_size": 8,
            "metric": "minkowski",
            "minkowski_p": 3.0,
            "clusters": 7,
            "noise": 0,
            "target": False,
        },
        id="minkowski_p3_eps1.0_min4_leaf8",
    ),
]


HDBSCAN_PARAMETER_CASES = [
    pytest.param(
        {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "max_cluster_size": 0,
            "alpha": 1.0,
            "algorithm": "auto",
            "leaf_size": 8,
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="auto_mcs5_ms5_eom",
    ),
    pytest.param(
        {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "max_cluster_size": 0,
            "alpha": 1.0,
            "algorithm": "brute",
            "leaf_size": 8,
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="brute_mcs5_ms5_eom",
    ),
    pytest.param(
        {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "max_cluster_size": 0,
            "alpha": 1.0,
            "algorithm": "kd_tree",
            "leaf_size": 8,
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="kd_tree_mcs5_ms5_eom",
    ),
    pytest.param(
        {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "max_cluster_size": 0,
            "alpha": 1.0,
            "algorithm": "ball_tree",
            "leaf_size": 8,
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="ball_tree_mcs5_ms5_eom",
    ),
    pytest.param(
        {
            "min_cluster_size": 4,
            "min_samples": 4,
            "cluster_selection_epsilon": 0.0,
            "max_cluster_size": 0,
            "alpha": 1.0,
            "algorithm": "brute",
            "leaf_size": 8,
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="brute_mcs4_ms4_eom",
    ),
    pytest.param(
        {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.2,
            "max_cluster_size": 1000,
            "alpha": 1.0,
            "algorithm": "ball_tree",
            "leaf_size": 12,
            "cluster_selection_method": "eom",
            "allow_single_cluster": True,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="ball_tree_mcs5_ms5_eps0.2_max1000_single",
    ),
    pytest.param(
        {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "max_cluster_size": 30,
            "alpha": 1.0,
            "algorithm": "brute",
            "leaf_size": 8,
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "clusters": 8,
            "noise": 9,
            "target": False,
        },
        id="brute_mcs5_ms5_max30_eom",
    ),
    pytest.param(
        {
            "min_cluster_size": 5,
            "min_samples": 0,
            "cluster_selection_epsilon": 0.0,
            "max_cluster_size": 0,
            "alpha": 1.0,
            "algorithm": "auto",
            "leaf_size": 8,
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "clusters": 7,
            "noise": 0,
            "target": True,
        },
        id="auto_mcs5_default_ms_eom",
    ),
    pytest.param(
        {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "max_cluster_size": 0,
            "alpha": 1.0,
            "algorithm": "kd_tree",
            "leaf_size": 8,
            "metric": "default",
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "clusters": 6,
            "noise": 11,
            "target": False,
        },
        id="default_cosine_mcs5_ms5_eom",
    ),
    pytest.param(
        {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "max_cluster_size": 0,
            "alpha": 1.0,
            "algorithm": "brute",
            "leaf_size": 8,
            "metric": "manhattan",
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "clusters": 7,
            "noise": 0,
            "target": False,
        },
        id="manhattan_mcs5_ms5_eom",
    ),
    pytest.param(
        {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "max_cluster_size": 0,
            "alpha": 1.0,
            "algorithm": "auto",
            "leaf_size": 8,
            "metric": "minkowski",
            "minkowski_p": 3.0,
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "clusters": 7,
            "noise": 0,
            "target": False,
        },
        id="minkowski_p3_mcs5_ms5_eom",
    ),
]


@pytest.mark.parametrize("case", DBSCAN_PARAMETER_CASES)
def test_dbscan_matches_sklearn_on_asset_data(ops_module, repo_root: Path, case) -> None:
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    samples, target_labels = _load_cluster_data(repo_root)
    algorithm, sklearn_algorithm = _algorithm_case(ops_module, case["algorithm"])
    metric, sklearn_metric = _metric_case(ops_module, case.get("metric", "euclidean"))
    config = ops_module.DBSCANConfig()
    config.eps = case["eps"]
    config.min_samples = case["min_samples"]
    config.algorithm = algorithm
    config.leaf_size = case["leaf_size"]
    if metric is not None:
        config.metric = metric
    if "minkowski_p" in case:
        config.minkowski_p = case["minkowski_p"]

    sklearn_kwargs = {
        "eps": config.eps,
        "min_samples": config.min_samples,
        "metric": sklearn_metric,
        "algorithm": _sklearn_algorithm_for_metric(sklearn_algorithm, sklearn_metric),
        "leaf_size": config.leaf_size,
    }
    if sklearn_metric == "minkowski":
        sklearn_kwargs["p"] = config.minkowski_p
    result = ops_module.dbscan(samples, config)
    expected = sklearn_cluster.DBSCAN(**sklearn_kwargs).fit_predict(samples)

    labels = np.asarray(result.labels, dtype=np.int64)
    np.testing.assert_array_equal(labels, expected)
    assert _count_clusters(labels) == case["clusters"]
    assert int(np.count_nonzero(labels == -1)) == case["noise"]
    if case["target"]:
        _assert_same_partition(labels, target_labels)


@pytest.mark.parametrize("case", HDBSCAN_PARAMETER_CASES)
def test_hdbscan_matches_sklearn_on_asset_data(ops_module, repo_root: Path, case) -> None:
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    if not hasattr(sklearn_cluster, "HDBSCAN"):
        pytest.skip("sklearn.cluster.HDBSCAN is not available")

    samples, target_labels = _load_cluster_data(repo_root)
    algorithm, sklearn_algorithm = _algorithm_case(ops_module, case["algorithm"])
    metric, sklearn_metric = _metric_case(ops_module, case.get("metric", "euclidean"))
    cluster_selection_method, sklearn_cluster_selection_method = _cluster_selection_method(
        ops_module,
        case["cluster_selection_method"],
    )
    config = ops_module.HDBSCANConfig()
    config.min_cluster_size = case["min_cluster_size"]
    config.min_samples = case["min_samples"]
    config.cluster_selection_epsilon = case["cluster_selection_epsilon"]
    config.max_cluster_size = case["max_cluster_size"]
    config.alpha = case["alpha"]
    config.algorithm = algorithm
    config.leaf_size = case["leaf_size"]
    if metric is not None:
        config.metric = metric
    if "minkowski_p" in case:
        config.minkowski_p = case["minkowski_p"]
    config.cluster_selection_method = cluster_selection_method
    config.allow_single_cluster = case["allow_single_cluster"]

    metric_params = {"p": config.minkowski_p} if sklearn_metric == "minkowski" else None
    result = ops_module.hdbscan(samples, config)
    reference = sklearn_cluster.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=None if config.min_samples == 0 else config.min_samples,
        cluster_selection_epsilon=config.cluster_selection_epsilon,
        max_cluster_size=None if config.max_cluster_size == 0 else config.max_cluster_size,
        metric=sklearn_metric,
        metric_params=metric_params,
        alpha=config.alpha,
        algorithm=_sklearn_algorithm_for_metric(sklearn_algorithm, sklearn_metric),
        leaf_size=config.leaf_size,
        cluster_selection_method=sklearn_cluster_selection_method,
        allow_single_cluster=config.allow_single_cluster,
        copy=True,
    ).fit(samples)

    labels = np.asarray(result.labels, dtype=np.int64)
    probabilities = np.asarray(result.probabilities, dtype=np.float64)
    np.testing.assert_array_equal(labels, reference.labels_)
    np.testing.assert_allclose(probabilities, reference.probabilities_)
    assert _count_clusters(labels) == case["clusters"]
    assert int(np.count_nonzero(labels == -1)) == case["noise"]
    assert len(result.probabilities) == len(result.labels)
    assert all(0.0 <= probability <= 1.0 for probability in result.probabilities)
    if case["target"]:
        _assert_same_partition(labels, target_labels)


def test_clustering_default_metric_is_cosine(ops_module) -> None:
    assert ops_module.DBSCANConfig().metric == ops_module.ClusteringMetric.Cosine
    assert ops_module.HDBSCANConfig().metric == ops_module.ClusteringMetric.Cosine


def test_clustering_rejects_invalid_sample_shape(ops_module) -> None:
    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.dbscan(np.zeros((2, 3, 1), dtype=np.float32), ops_module.DBSCANConfig())

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.hdbscan(np.zeros((2, 3, 1), dtype=np.float32), ops_module.HDBSCANConfig())

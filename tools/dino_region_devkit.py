#!/usr/bin/env python
"""DINO 区域检索的开发集夹具与阈值标定工具。

工具独立于检索实现，所有文本输入输出使用 YAML（需要 PyYAML、NumPy、OpenCV）：

- ``generate``：用真实照片与确定性几何变换构造开发图库、标签与基准清单。
  每个图库图片都是某个已标注目标的**上下文裁剪**经缩放/旋转后的结果，
  因此正例的边界框由同一变换精确映射得到，不依赖人工重标。
- ``calibrate``：在开发集上扫描评分权重与最终判定阈值。
- ``probe``：同图定位、未标注留出集返回率与阶段耗时。

几何夹具证明同一实例的变换对应，不证明类别语义或独立真实图的检索质量。
未知背景不是已确认负例；真实质量需要单独人工标注的独立查询。
"""

from __future__ import annotations

import argparse
import yaml
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- 开发集定义

#: 已标注目标：来源照片 + canonical 像素边界框 ``[x0, y0, x1, y1)``。
#: 名称只是夹具实例标识，不是经审核的类别相关性标签。
DEFAULT_OBJECTS: tuple[dict, ...] = (
    {"name": "dog", "source": "assets/pics/dog.jpg", "box": [128, 222, 322, 540]},
    {"name": "thrust_bearing", "source": "assets/pics/shape1.jpg", "box": [420, 595, 547, 719]},
    {"name": "brass_washer", "source": "assets/pics/shape1.jpg", "box": [1538, 339, 1621, 425]},
    {"name": "hex_nut", "source": "assets/pics/shape1.jpg", "box": [1198, 254, 1298, 350]},
)

#: 不入图库的来源照片；没有人工相关性标注，不能据此认定无匹配。
DEFAULT_HOLDOUT_SOURCE = "assets/pics/bus.jpg"

#: 未标注留出查询区域（来源照片像素）。
DEFAULT_HOLDOUT_BOXES: tuple[tuple[int, int, int, int], ...] = (
    (60, 300, 300, 620),
    (380, 250, 700, 460),
    (100, 850, 500, 1050),
)

#: 上下文裁剪相对目标框的扩边倍率。
CONTEXT_EXPAND = 1.6

#: 几何变换网格：``(scale, rotation_degrees)``。每个目标生成一份副本。
TRANSFORM_GRID: tuple[tuple[float, float], ...] = (
    (0.75, -15.0),
    (0.75, 0.0),
    (0.75, 15.0),
    (1.0, -15.0),
    (1.0, 0.0),
    (1.0, 15.0),
    (1.5, -15.0),
    (1.5, 0.0),
    (1.5, 15.0),
    (2.0, 0.0),
)

#: 标准验收承诺的最小目标短边；低于它的副本不作为正例。
MIN_TARGET_SHORT_PX = 64.0

JPEG_QUALITY = 92


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def clamp(self, width: int, height: int) -> "Rect":
        return Rect(
            max(0.0, min(self.x0, width)),
            max(0.0, min(self.y0, height)),
            max(0.0, min(self.x1, width)),
            max(0.0, min(self.y1, height)),
        )

    def as_list(self) -> list[float]:
        return [round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2)]


def expand(rect: Rect, factor: float, width: int, height: int) -> Rect:
    center_x = (rect.x0 + rect.x1) * 0.5
    center_y = (rect.y0 + rect.y1) * 0.5
    half_w = rect.width * factor * 0.5
    half_h = rect.height * factor * 0.5
    return Rect(center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h).clamp(width, height)


def similarity_matrix(scale: float, angle_degrees: float, crop: Rect) -> np.ndarray:
    """把上下文裁剪映射到输出图：绕裁剪中心缩放并旋转，再平移到输出左上角。"""
    radians = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    center_x = crop.width * 0.5
    center_y = crop.height * 0.5
    output_center_x = center_x * scale
    output_center_y = center_y * scale
    linear = np.array([[scale * cos_a, -scale * sin_a], [scale * sin_a, scale * cos_a]], dtype=np.float32)
    translation = np.array([output_center_x, output_center_y], dtype=np.float32) - linear @ np.array(
        [center_x, center_y], dtype=np.float32
    )
    return np.hstack([linear, translation.reshape(2, 1)]).astype(np.float32)


def transform_rect(matrix: np.ndarray, rect: Rect) -> Rect:
    corners = np.array(
        [[rect.x0, rect.y0], [rect.x1, rect.y0], [rect.x1, rect.y1], [rect.x0, rect.y1]], dtype=np.float32
    )
    mapped = (matrix[:, :2] @ corners.T).T + matrix[:, 2]
    return Rect(float(mapped[:, 0].min()), float(mapped[:, 1].min()), float(mapped[:, 0].max()), float(mapped[:, 1].max()))


def load_object_specs(path: Path | None) -> list[dict]:
    if path is None:
        return [dict(item) for item in DEFAULT_OBJECTS]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    objects = payload["objects"] if isinstance(payload, dict) else payload
    return [dict(item) for item in objects]


def render_copy(image: np.ndarray, crop: Rect, matrix: np.ndarray, output_path: Path) -> None:
    """按给定变换渲染一份上下文副本。"""
    x0, y0 = int(round(crop.x0)), int(round(crop.y0))
    x1, y1 = int(round(crop.x1)), int(round(crop.y1))
    patch = image[y0:y1, x0:x1]
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    out_w = max(8, int(round(crop.width * scale)))
    out_h = max(8, int(round(crop.height * scale)))
    warped = cv2.warpAffine(
        patch, matrix, (out_w, out_h), flags=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded, buffer = cv2.imencode(".jpg", warped, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not encoded:
        raise SystemExit(f"cannot encode image: {output_path}")
    buffer.tofile(str(output_path))


def generate(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out_dir = Path(args.out)
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    gallery_dir = out_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    objects = load_object_specs(Path(args.objects) if args.objects else None)
    sources: dict[str, np.ndarray] = {}
    for spec in objects:
        source = str(spec["source"])
        if source not in sources:
            sources[source] = decode_image(root / source)

    gallery_entries: list[dict] = []
    positive_boxes: dict[str, list[dict]] = {spec["name"]: [] for spec in objects}

    for spec in objects:
        name = spec["name"]
        source = str(spec["source"])
        image = sources[source]
        height, width = image.shape[:2]
        object_box = Rect(*[float(v) for v in spec["box"]])
        crop = expand(object_box, CONTEXT_EXPAND, width, height)
        # 使用实际整数裁剪原点计算同一个仿射变换，避免亚像素标签偏移。
        crop = Rect(*(float(round(value)) for value in (crop.x0, crop.y0, crop.x1, crop.y1)))
        for scale, angle in TRANSFORM_GRID:
            matrix = similarity_matrix(scale, angle, crop)
            local_box = transform_rect(matrix, Rect(object_box.x0 - crop.x0, object_box.y0 - crop.y0,
                                                   object_box.x1 - crop.x0, object_box.y1 - crop.y0))
            out_w = max(8, int(round(crop.width * scale)))
            out_h = max(8, int(round(crop.height * scale)))
            visible = local_box.clamp(out_w, out_h)
            # 目标缩小到已验证范围之外、或者被裁掉一块的副本不进图库：
            # 否则图库中会存在“真实存在但无法标注”的目标，评标时会被误判成误检。
            if min(out_w, out_h) < 256 or min(visible.width, visible.height) < MIN_TARGET_SHORT_PX:
                continue
            if visible.area < local_box.area * 0.98:
                continue

            file_name = f"{name}__{scale:g}_{angle:+g}".replace("+", "p").replace("-", "m") + ".jpg"
            render_copy(image, crop, matrix, gallery_dir / file_name)
            gallery_entries.append(
                {
                    "name": file_name,
                    "object": name,
                    "source": source,
                    "scale": scale,
                    "angle_degrees": angle,
                    "box": visible.as_list(),
                }
            )
            positive_boxes[name].append({"image": file_name, "box": visible.as_list()})

    queries: list[dict] = []
    for spec in objects:
        name = spec["name"]
        if not positive_boxes[name]:
            raise SystemExit(f"object '{name}' produced no usable positive; adjust the transform grid")
        queries.append(
            {
                "query_id": f"dev-{name}",
                "query_path": str(root / str(spec["source"])),
                "include_self": False,
                "roi": {"bbox": [float(v) for v in spec["box"]]},
                "positives": [
                    {"image_path": str((gallery_dir / item["image"]).resolve()), "bbox": item["box"],
                     "relevance": "positive"}
                    for item in positive_boxes[name]
                ],
                "no_match": False,
            }
        )

    holdout_source = root / args.holdout_source
    decode_image(holdout_source)
    for index, box in enumerate(DEFAULT_HOLDOUT_BOXES):
        queries.append(
            {
                "query_id": f"dev-holdout-{index}",
                "query_path": str(holdout_source),
                "include_self": False,
                "roi": {"bbox": [float(v) for v in box]},
                "positives": [],
                "no_match": False,
                "full_annotation": False,
            }
        )

    manifest = {
        "mode": "partial_annotation",
        "note": (
            "Geometric same-instance positives only; object names are fixture identifiers, not "
            "category relevance labels. Unannotated background and holdout queries are not negatives. "
            "Independent annotated photos are required for real-world quality claims."
        ),
        "queries": queries,
    }
    manifest_path = out_dir / "benchmark_dev.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False) + "\n", encoding="utf-8")

    labels_path = out_dir / "labels_dev.yaml"
    labels_path.write_text(
        yaml.safe_dump({
            "objects": objects,
            "context_expand": CONTEXT_EXPAND,
            "transform_grid": TRANSFORM_GRID,
            "gallery": gallery_entries,
            "positive_count": {name: len(boxes) for name, boxes in positive_boxes.items()},
            "unlabelled_queries": [query["query_id"] for query in queries if not query["positives"]],
        }, allow_unicode=True, sort_keys=False)
        + "\n",
        encoding="utf-8",
    )

    print(yaml.safe_dump({
        "gallery_images": len(gallery_entries),
        "positive_boxes": {name: len(boxes) for name, boxes in positive_boxes.items()},
        "positive_queries": len(queries) - len(DEFAULT_HOLDOUT_BOXES),
        "unlabelled_queries": len(DEFAULT_HOLDOUT_BOXES),
        "manifest": str(manifest_path),
        "labels": str(labels_path),
    }, allow_unicode=True, sort_keys=False))
    return 0


def run_search(cli: Path, index: Path, profile: Path, request: dict, cache_dir: Path,
               extra_args: tuple = ()) -> dict:
    """把一条查询转换成 YAML 请求并调用 CLI；保留未完成状态供报告使用。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    request_path = cache_dir / f"{request['query_id']}.request.yaml"
    body = {
        "request_id": request["query_id"],
        "query_path": request["query_path"],
        **request["roi"],
        "include_self": request.get("include_self", False),
    }
    request_path.write_text(yaml.safe_dump(body, allow_unicode=True, sort_keys=False), encoding="utf-8")
    completed = subprocess.run(
        [str(cli), "search", "--index", str(index), "--profile", str(profile), "--request", str(request_path),
         *extra_args],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    # 退出码 5 表示查询未完成，此时仍然有部分结果的 YAML 响应。
    if completed.returncode not in (0, 5):
        raise SystemExit(f"search failed for {request['query_id']}: exit {completed.returncode}\n{completed.stderr}")
    return yaml.safe_load(completed.stdout)


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def coarse_coverage(positives: list[tuple[Path, list[float]]], candidates: list[dict]) -> int:
    """独立统计每个 GT 的候选覆盖；同一候选可以覆盖多个 GT。"""
    matched = 0
    for positive_path, gt in positives:
        gt_area = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
        if gt_area <= 0:
            raise ValueError("positive bbox must have nonzero area")
        for candidate in candidates:
            if Path(candidate["source_path"]).resolve() != positive_path:
                continue
            box = candidate["bbox"]
            intersection = max(0.0, min(gt[2], box[2]) - max(gt[0], box[0])) * max(0.0, min(gt[3], box[3]) - max(gt[1], box[1]))
            candidate_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            if intersection / gt_area >= 0.90 and candidate_area / gt_area <= 16.0:
                matched += 1
                break
    return matched


def evaluate_weights(cli: Path, index: Path, profile: Path, manifest: dict, cache_dir: Path,
                     iou_threshold: float) -> list[dict]:
    """按分数降序一对一匹配标注；重复命中为 FP，未标注背景不作负例。"""
    rows: list[dict] = []
    for query in manifest["queries"]:
        response = run_search(cli, index, profile, query, cache_dir)
        positives = [(Path(item["image_path"]).resolve(), item["bbox"]) for item in query.get("positives", [])]
        coarse = {channel: coarse_coverage(positives, response[field]) if field in response else None
                  for channel, field in (("region", "region_candidates"), ("local", "local_candidates"), ("fused", "coarse_candidates"))}
        no_match = query.get("no_match", False)
        exhaustive = query.get("full_annotation", manifest.get("mode") == "full_annotation")
        matched: set[int] = set()
        records = []
        for item in sorted(response["results"], key=lambda result: result["score"], reverse=True):
            path = Path(item["source_path"]).resolve()
            overlaps = [(index_positive, iou(item["bbox"], box))
                        for index_positive, (positive_path, box) in enumerate(positives) if positive_path == path]
            eligible = [(index_positive, overlap) for index_positive, overlap in overlaps
                        if index_positive not in matched and overlap >= iou_threshold]
            best_index, best_iou = max(eligible, key=lambda pair: pair[1], default=(-1, 0.0))
            is_true_positive = best_index >= 0
            duplicate = any(index_positive in matched and overlap >= iou_threshold
                            for index_positive, overlap in overlaps)
            if is_true_positive:
                matched.add(best_index)
            records.append({"image": str(path), "score": item["score"], "iou": best_iou,
                            "tp": is_true_positive, "fp": not is_true_positive and (duplicate or no_match or exhaustive)})
        rows.append({"query_id": query["query_id"], "no_match": no_match,
                     "status": response["status"], "positive_count": len(positives), "results": records,
                     "coarse_covered": coarse, "timings": response["timings"], "diagnostics": response.get("diagnostics", {})})
    return rows


def sweep_thresholds(rows: list[dict], thresholds: list[float]) -> list[dict]:
    evaluations = []
    for threshold in thresholds:
        tp = fp = fn = 0
        no_match_false_positive = 0
        no_match_queries = 0
        for row in rows:
            kept = [record for record in row["results"] if record["score"] >= threshold]
            row_tp = sum(1 for record in kept if record["tp"])
            tp += row_tp
            fp += sum(1 for record in kept if record["fp"])
            fn += row["positive_count"] - row_tp
            if row["no_match"]:
                no_match_queries += 1
                if kept:
                    no_match_false_positive += 1
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        evaluations.append(
            {
                "threshold": threshold,
                "micro_precision": precision,
                "micro_recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "no_match_fpr": no_match_false_positive / no_match_queries if no_match_queries else float("nan"),
                "no_match_queries": no_match_queries,
                "unknown_results": sum(1 for row in rows for record in row["results"]
                                       if record["score"] >= threshold and not record["tp"] and not record["fp"]),
            }
        )
    return evaluations


def threshold_candidates(rows: list[dict]) -> list[float]:
    scores = sorted({record["score"] for row in rows for record in row["results"]})
    if not scores:
        return [0.0]
    return [0.0] + [round((a + b) * 0.5, 6) for a, b in zip(scores, scores[1:])] + [round(scores[-1] + 1e-6, 6)]


def write_weight_profile(profile_path: Path, weights: tuple[float, float, float], out_path: Path) -> Path:
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile.setdefault("fine", {})["score_weights"] = list(weights)
    profile.setdefault("decision", {})["threshold"] = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False) + "\n", encoding="utf-8")
    return out_path


def calibrate(args: argparse.Namespace) -> int:
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    cli = Path(args.cli)
    index = Path(args.index)
    profile = Path(args.profile)
    cache_dir = Path(args.out).parent / "calibration_requests"
    report_path = Path(args.out)

    weight_grid = [tuple(item) for item in yaml.safe_load(Path(args.weights).read_text(encoding="utf-8"))] \
        if args.weights else [(0.60, 0.25, 0.15)]

    grid: list[dict] = []
    for weights in weight_grid:
        profile_variant = write_weight_profile(profile, weights, cache_dir / f"profile_{'_'.join(f'{w:g}' for w in weights)}.yaml")
        rows = evaluate_weights(cli, index, profile_variant, manifest, cache_dir, args.iou_threshold)
        evaluations = sweep_thresholds(rows, threshold_candidates(rows))
        feasible = [
            item for item in evaluations
            if item["micro_precision"] >= args.min_precision
            and item["micro_recall"] >= args.min_recall
            and (item["no_match_queries"] == 0 or item["no_match_fpr"] <= args.max_no_match_fpr)
            and all(row["status"] == "completed" for row in rows)
        ]
        best = max(evaluations, key=lambda item: (min(item["micro_precision"], item["micro_recall"]),
                                                  item["micro_recall"]))
        grid.append(
            {
                "score_weights": list(weights),
                "profile": str(profile_variant),
                "feasible_thresholds": [item["threshold"] for item in feasible],
                "selected": max(feasible, key=lambda item: (item["micro_recall"], -item["threshold"])) if feasible else None,
                "best_operating_point": best,
                "queries": rows,
                "coarse": {
                    "known_positives": sum(row["positive_count"] for row in rows),
                    "min_gt_coverage": 0.90,
                    "max_area_ratio": 16.0,
                    "recall": {channel: (sum(row["coarse_covered"][channel] for row in rows)
                                         / sum(row["positive_count"] for row in rows))
                               if all(row["coarse_covered"][channel] is not None for row in rows)
                               and sum(row["positive_count"] for row in rows) else None
                               for channel in ("region", "local", "fused")},
                },
                "sweep": evaluations,
            }
        )

    feasible_entries = [entry for entry in grid if entry["selected"] is not None]
    chosen = max(feasible_entries, key=lambda entry: entry["selected"]["micro_recall"]) if feasible_entries else None

    report = {
        "development_set": str(Path(args.manifest)),
        "iou_threshold": args.iou_threshold,
        "evidence": "Geometric instance correspondence only; not category-level relevance or blind-set quality.",
        "no_match_fpr_verified": any(query.get("no_match", False) for query in manifest["queries"]),
        "constraints": {
            "min_precision": args.min_precision,
            "min_recall": args.min_recall,
            "max_no_match_fpr": args.max_no_match_fpr,
        },
        "grid": grid,
        "selected": None if chosen is None else {"score_weights": chosen["score_weights"], **chosen["selected"]},
    }
    report_path.write_text(yaml.safe_dump(report, allow_unicode=True, sort_keys=False) + "\n", encoding="utf-8")

    summary = {
        "selected": report["selected"],
        "operating_points": [
            {
                "score_weights": entry["score_weights"],
                "precision": entry["best_operating_point"]["micro_precision"],
                "recall": entry["best_operating_point"]["micro_recall"],
                "no_match_fpr": entry["best_operating_point"]["no_match_fpr"],
            }
            for entry in grid
        ],
        "report": str(report_path),
    }
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    return 0 if chosen is not None else 1


# ---------------------------------------------------------------- 真实数据探测

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def decode_image(path: Path):
    """读真实数据集里的照片。

    OpenCV 的 ``imread`` 在 Windows 上按窄字符打开文件，中文路径会失败；
    这里先用 ``np.fromfile`` 读字节再解码，与 CLI 内部的处理方式一致。
    """
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot decode image: {path}")
    return image


def center_roi(path: Path, roi_edge: int) -> list[float]:
    """取图像中心的方形 ROI；不依赖任何标注，用于自查询定位自检与无匹配探测。"""
    shape = decode_image(path).shape
    height, width = shape[0], shape[1]
    edge = float(min(roi_edge, width, height))
    x0 = (float(width) - edge) * 0.5
    y0 = (float(height) - edge) * 0.5
    return [x0, y0, x0 + edge, y0 + edge]


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    """确定性地在有序列表上取均匀样本，避免依赖随机数状态。"""
    if count <= 0 or not paths:
        return []
    if len(paths) <= count:
        return list(paths)
    step = len(paths) / float(count)
    return [paths[int(index * step)] for index in range(count)]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def median(values: list[float]) -> float:
    return percentile(values, 0.5) if values else float("nan")


def probe(args: argparse.Namespace) -> int:
    """探测同图定位、未标注 holdout 返回率与阶段耗时；不推断无匹配。"""
    cli, index, profile = Path(args.cli), Path(args.index), Path(args.profile)
    cache_dir = Path(args.out).parent / "probe_requests"

    gallery_images = sorted(path for path in Path(args.gallery_root).rglob("*")
                            if path.suffix.lower() in IMAGE_SUFFIXES)
    holdout_images: list[Path] = []
    for root in [item for item in args.holdout.split(",") if item]:
        holdout_images.extend(sorted(path for path in Path(root).rglob("*")
                                     if path.suffix.lower() in IMAGE_SUFFIXES))

    extra = []
    if args.threshold is not None:
        probe_profile = yaml.safe_load(profile.read_text(encoding="utf-8"))
        probe_profile.setdefault("decision", {})["threshold"] = args.threshold
        cache_dir.mkdir(parents=True, exist_ok=True)
        profile = cache_dir / "profile.yaml"
        profile.write_text(yaml.safe_dump(probe_profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if args.deadline_ms is not None:
        # 只为取得完整的阶段耗时；冻结 profile 的截止时间本身不修改。
        extra += ["--deadline-ms", str(args.deadline_ms)]

    cases = [{"query_id": f"self-{index:02d}", "image": path, "expect": "self"}
             for index, path in enumerate(evenly_spaced(gallery_images, args.count))]
    cases += [{"query_id": f"holdout-{index:02d}", "image": path, "expect": "unlabelled"}
              for index, path in enumerate(evenly_spaced(holdout_images, args.count))]

    rows: list[dict] = []
    for case in cases:
        roi = center_roi(Path(case["image"]), args.roi)
        request = {"query_id": case["query_id"], "query_path": str(case["image"]),
                   "include_self": case["expect"] == "self",
                   "roi": {"bbox": roi},
                   "positives": []}
        response = run_search(cli, index, profile, request, cache_dir, extra)
        results = response.get("results", []) or []
        counters = response.get("diagnostics", {}) or {}
        rows.append(
            {
                "query_id": case["query_id"],
                "expect": case["expect"],
                "image": str(case["image"]),
                "roi": roi,
                "returned": len(results),
                "top_score": max((item["score"] for item in results), default=0.0),
                "best_iou": max((iou(item["bbox"], roi) for item in results
                                 if Path(item["source_path"]).resolve() == Path(case["image"]).resolve()), default=0.0),
                "status": response["status"],
                "decision": response["decision"],
                "timings": response["timings"],
                "scanned_region_descriptors": counters.get("scanned_region_descriptors", 0),
                "scanned_local_descriptors": counters.get("scanned_local_descriptors", 0),
                "view_count": counters.get("view_count", 0),
                "model_forwards": counters.get("model_forwards", 0),
            }
        )

    self_rows = [row for row in rows if row["expect"] == "self"]
    holdout_rows = [row for row in rows if row["expect"] == "unlabelled"]
    walls = [row["timings"]["wall_ms"] for row in rows]
    stage_names = ["decode_ms", "query_extract_ms", "region_scan_ms", "local_scan_ms",
                   "local_window_rescore_ms", "fusion_ms", "fine_extract_ms", "fine_match_ms"]

    report = {
        "gallery_root": str(args.gallery_root),
        "holdout": args.holdout,
        "roi": args.roi,
        "threshold": args.threshold,
        "deadline_ms_override": args.deadline_ms,
        "incomplete_queries": sum(1 for row in rows if row["status"] != "completed"),
        "index_view_count": self_rows[0]["view_count"] if self_rows else None,
        "self_localisation": {
            "queries": len(self_rows),
            "hit_at_iou_0_5": sum(1 for row in self_rows if row["best_iou"] >= 0.5),
            "median_best_iou": median([row["best_iou"] for row in self_rows]),
            "min_best_iou": min((row["best_iou"] for row in self_rows), default=float("nan")),
            "median_top_score": median([row["top_score"] for row in self_rows]),
        },
        "unlabelled_holdout": {
            "queries": len(holdout_rows),
            "returned_queries": sum(1 for row in holdout_rows if row["returned"] > 0),
            "return_rate": (sum(1 for row in holdout_rows if row["returned"] > 0)
                            / len(holdout_rows)) if holdout_rows else float("nan"),
            "max_top_score": max((row["top_score"] for row in holdout_rows), default=float("nan")),
            "evidence": "Not annotated: no true negatives or false-positive rate can be inferred.",
        },
        "timings_ms": {
            "p50_wall": median(walls),
            "p95_wall": percentile(walls, 0.95),
            "max_wall": max(walls) if walls else float("nan"),
            "median_stage": {name: median([row["timings"][name] for row in rows]) for name in stage_names},
        },
        "scanned": {
            "median_region_descriptors": median([row["scanned_region_descriptors"] for row in rows]),
            "median_local_descriptors": median([row["scanned_local_descriptors"] for row in rows]),
            "median_model_forwards": median([row["model_forwards"] for row in rows]),
        },
        "queries": rows,
    }
    Path(args.out).write_text(yaml.safe_dump(report, allow_unicode=True, sort_keys=False) + "\n", encoding="utf-8")
    print(yaml.safe_dump({key: report[key] for key in
                      ["threshold", "deadline_ms_override", "incomplete_queries", "index_view_count",
                       "self_localisation", "unlabelled_holdout", "timings_ms", "scanned"]}, allow_unicode=True, sort_keys=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    generate_parser = sub.add_parser("generate", help="build the development gallery and benchmark manifest")
    generate_parser.add_argument("--root", default=".", help="repository root used to resolve source photos")
    generate_parser.add_argument("--out", required=True, help="output directory for gallery, labels and manifest")
    generate_parser.add_argument("--objects", default=None, help="optional YAML file overriding the object list")
    generate_parser.add_argument("--holdout-source", default=DEFAULT_HOLDOUT_SOURCE,
                                 help="unannotated photo excluded from the gallery; not presumed negative")
    generate_parser.add_argument("--clean", action="store_true", help="remove the output directory first")
    generate_parser.set_defaults(func=generate)

    calibrate_parser = sub.add_parser("calibrate", help="sweep the final decision threshold on the dev set")
    calibrate_parser.add_argument("--manifest", required=True)
    calibrate_parser.add_argument("--index", required=True)
    calibrate_parser.add_argument("--profile", required=True)
    calibrate_parser.add_argument("--cli", required=True)
    calibrate_parser.add_argument("--out", required=True)
    calibrate_parser.add_argument("--weights", default=None,
                                  help="YAML list of [template, coverage, consistency] triples to sweep")
    calibrate_parser.add_argument("--iou-threshold", type=float, default=0.5)
    calibrate_parser.add_argument("--min-precision", type=float, default=0.90)
    calibrate_parser.add_argument("--min-recall", type=float, default=0.90)
    calibrate_parser.add_argument("--max-no-match-fpr", type=float, default=0.05)
    calibrate_parser.set_defaults(func=calibrate)

    probe_parser = sub.add_parser("probe", help="probe a real gallery: self-localisation, unlabelled holdout returns and timings")
    probe_parser.add_argument("--gallery-root", required=True,
                              help="gallery directory already indexed; sampled for self-query localisation checks")
    probe_parser.add_argument("--holdout", required=True,
                              help="comma separated directories excluded from the gallery; not presumed negative")
    probe_parser.add_argument("--index", required=True)
    probe_parser.add_argument("--profile", required=True)
    probe_parser.add_argument("--cli", required=True)
    probe_parser.add_argument("--out", required=True)
    probe_parser.add_argument("--count", type=int, default=8, help="queries sampled per group")
    probe_parser.add_argument("--roi", type=int, default=512, help="square ROI edge in canonical pixels")
    probe_parser.add_argument("--threshold", type=float, default=None,
                              help="override the decision threshold in a temporary YAML profile")
    probe_parser.add_argument("--deadline-ms", type=int, default=None,
                              help="override the profile query deadline to obtain complete stage timings")
    probe_parser.set_defaults(func=probe)

    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

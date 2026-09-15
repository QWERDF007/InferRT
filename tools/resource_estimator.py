#!/usr/bin/env python3
"""Storage and operation counts from the implemented layout, not measured latency."""
import argparse
import json


def axis(length, edge, overlap=.25):
    tile = min(length, edge)
    stride = max(1, int(edge * (1 - overlap) + .5))
    points = list(range(0, length - tile + 1, stride))
    if not points or points[-1] != length - tile:
        points.append(length - tile)
    return points, tile


def view_count(width, height, edges=(512, 1024, 2048)):
    boxes = {(0, 0, width, height)}
    for edge in edges:
        xs, w = axis(width, edge)
        ys, h = axis(height, edge)
        boxes.update((x, y, x + w, y + h) for y in ys for x in xs)
    return len(boxes)


def estimate(width, height, images, dimension=96, representatives=64, query_tokens=64):
    views = view_count(width, height) * images
    # Existing packed metadata:32B, two float scales:8B, codes:d B.
    # Existing view table:18 doubles. Two offset tables:8B*(views+1) each.
    descriptor_bytes = views * (representatives + 10) * (dimension + 40)
    view_bytes = views * 144 + 16 * (views + 1)
    return {'kind': 'upper_bound_for_normal_compact_profile_excluding_headers_yaml_images_model',
            'images': images, 'width': width, 'height': height, 'views_per_image': view_count(width, height),
            'views': views, 'coarse_dimension': dimension, 'representatives': representatives,
            'index_payload_bytes_upper_bound': descriptor_bytes + view_bytes,
            'index_payload_gib_upper_bound': (descriptor_bytes + view_bytes) / 2 ** 30,
            'local_scan_macs': views * representatives * query_tokens * dimension,
            'local_pair_download_bytes': views * query_tokens * 16,
            'maximum_votes_per_view_per_query_context': (query_tokens // 2) * 2 * 3,
            'full_dense_gallery_features_persisted': False,
            'end_to_end_latency_ms': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=3060)
    parser.add_argument('--height', type=int, default=4520)
    parser.add_argument('--images', type=int, default=5000)
    parser.add_argument('--dimension', type=int, default=96)
    parser.add_argument('--representatives', type=int, default=64)
    parser.add_argument('--query-tokens', type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(estimate(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Stage recall, ranking, localization and resources. Uses PyYAML; no trained component."""
import argparse
import math
from pathlib import Path
import statistics
import yaml


def path_key(value):
    # GT uses the same canonical paths as responses. Keep Linux case sensitivity.
    return str(value).replace('\\', '/')


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    z = overlap(a, b)
    u = area(a) + area(b) - z
    return z / u if u > 0 else 0.0


def same_image(a, b):
    return path_key(a['source_path']) == path_key(b['source_path'])


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    at = (len(values) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def match_targets(predictions, targets, threshold):
    # Maximum bipartite matching prevents one large detection from counting twice.
    links = [[j for j, t in enumerate(targets) if same_image(p, t) and iou(p['bbox'], t['bbox']) >= threshold]
             for p in predictions]
    assigned = {}

    def augment(p, seen):
        for t in links[p]:
            if t in seen:
                continue
            seen.add(t)
            if t not in assigned or augment(assigned[t], seen):
                assigned[t] = p
                return True
        return False
    for p in range(len(predictions)):
        augment(p, set())
    return assigned


def evaluate(annotations, responses, k=20, threshold=0.5):
    indexed = {}
    for response in responses:
        qid = response['request_id']
        if qid in indexed:
            raise ValueError(f'duplicate response for {qid}; select one measured run')
        indexed[qid] = response
    rows = []
    for query in annotations['queries']:
        qid = query['request_id']
        targets = query['targets']
        for target in targets:
            if not isinstance(target.get('bbox'), list) or len(target['bbox']) != 4 or area(target['bbox']) <= 0:
                raise ValueError(f'{qid}: missing usable ground truth bbox')
        response = indexed.get(qid, {'status': 'missing', 'results': []})
        raw = response.get('results') or []
        # First truncate the raw output, then remove the query's own ROI; never backfill ranks.
        def eligible(p):
            own = query.get('exclude_roi')
            return not (own and same_image(p, own) and iou(p['bbox'], own['bbox']) >= 0.5)
        predictions = [p for p in raw[:k] if eligible(p)]
        found = match_targets(predictions, targets, threshold)
        top3 = [p for p in raw[:3] if eligible(p)]
        first_rank = next((i + 1 for i, p in enumerate(raw[:k]) if eligible(p) and match_targets([p], targets, threshold)), None)
        row = {'request_id': qid, 'tags': query.get('tags', []), 'status': response['status'],
               'positive_pairs': len(targets), 'retrieved_pairs': len(found), 'first_hit_rank': first_rank,
               'hit_at_3': bool(match_targets(top3, targets, threshold)),
               'decision': response.get('decision', 'unknown'), 'predictions_at_k': len(predictions), 'recall_at_k': len(found) / len(targets) if targets else None}
        for stage in ['region_candidates', 'local_candidates', 'coarse_candidates', 'localized_candidates', 'verification_candidates']:
            if stage not in response:
                row[stage + '_hits'] = None
                continue
            candidates = response.get(stage) or []
            if stage in ('localized_candidates', 'verification_candidates'):
                hits = len(match_targets(candidates, targets, threshold))
            else:
                hits = sum(any(same_image(p, t) and overlap(p['bbox'], t['bbox']) / area(t['bbox']) >= .9
                               and area(p['bbox']) / area(t['bbox']) <= 16 for p in candidates) for t in targets)
            row[stage + '_hits'] = hits
        row['best_iou_per_target'] = [max((iou(p['bbox'], t['bbox']) for p in predictions if same_image(p, t)), default=0)
                                      for t in targets]
        row['wall_ms'] = response.get('timings', {}).get('wall_ms')
        row['peak_rss_bytes'] = response.get('diagnostics', {}).get('peak_rss_bytes')
        # Precision can only be interpreted with exhaustive labels in the evaluated search universe.
        row['precision_at_returned_k'] = len(found) / len(predictions) if annotations.get('exhaustive_labels', False) and predictions else None
        rows.append(row)

    def aggregate(items):
        pairs = sum(x['positive_pairs'] for x in items)
        positives = [x for x in items if x['positive_pairs']]
        latency = [x['wall_ms'] for x in items if x['wall_ms'] is not None]
        stages = {}
        for name in ['region_candidates', 'local_candidates', 'coarse_candidates', 'localized_candidates', 'verification_candidates']:
            values = [x[name + '_hits'] for x in items if x['positive_pairs']]
            stages[name + '_recall'] = sum(values) / pairs if pairs and all(v is not None for v in values) else None
        negatives = [x for x in items if x['positive_pairs'] == 0]
        decidable_negatives = [x for x in negatives if x['status'] == 'completed' and x['decision'] in ('matches', 'no_match')]
        precisions = [x for x in items if x['precision_at_returned_k'] is not None]
        return {'negative_queries': len(negatives),
                'decidable_negative_queries': len(decidable_negatives),
                'negative_false_match_rate': sum(x['decision'] == 'matches' for x in decidable_negatives) / len(decidable_negatives) if decidable_negatives and annotations.get('exhaustive_labels', False) else None,
                'micro_precision_at_returned_k': sum(x['retrieved_pairs'] for x in precisions) / sum(x['predictions_at_k'] for x in precisions) if precisions else None,
                'queries': len(items), 'completed': sum(x['status'] == 'completed' for x in items),
                'positive_pairs': pairs, 'recall_at_k': sum(x['retrieved_pairs'] for x in items) / pairs if pairs else None,
                'query_hit_at_3': sum(x['hit_at_3'] for x in positives) / len(positives) if positives else None,
                'mrr_at_k': statistics.mean(1 / x['first_hit_rank'] if x['first_hit_rank'] else 0 for x in positives) if positives else None,
                'wall_p50_ms': percentile(latency, .5), 'wall_p95_ms': percentile(latency, .95),
                'incomplete_or_missing': sum(x['status'] != 'completed' for x in items), **stages}
    tags = sorted({t for row in rows for t in row['tags']})
    return {'measurement': 'provided_response_files', 'k': k, 'iou_threshold': threshold,
            'coarse_metric': 'geometric_crop_proxy_coverage_0.9_area_ratio_le16',
            'overall': aggregate(rows), 'by_tag': {t: aggregate([x for x in rows if t in x['tags']]) for t in tags}, 'queries': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--responses', nargs='+', type=Path, required=True, help='single or multi-document response YAML files')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--iou', type=float, default=.5)
    args = parser.parse_args()
    responses = []
    for path in args.responses:
        responses.extend(x for x in yaml.safe_load_all(path.read_text(encoding='utf-8')) if x)
    report = evaluate(yaml.safe_load(args.annotations.read_text(encoding='utf-8')), responses, args.k, args.iou)
    args.output.write_text(yaml.safe_dump(report, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(yaml.safe_dump(report['overall'], sort_keys=False))


if __name__ == '__main__':
    main()

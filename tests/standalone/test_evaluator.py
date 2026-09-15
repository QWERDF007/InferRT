import importlib.util
from pathlib import Path
import unittest
path = Path(__file__).resolve().parents[2] / 'tools/evaluate_region_search.py'
spec = importlib.util.spec_from_file_location('evaluation', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class EvaluationTests(unittest.TestCase):
    def p(self, box, score=1):
        return {'source_path': '/gallery/a.png', 'bbox': box, 'score': score}
    def test_raw_rank_no_backfill(self):
        own = self.p([0, 0, 10, 10]); target = self.p([20, 0, 30, 10])
        gt = {'queries': [{'request_id': 'q', 'targets': [target], 'exclude_roi': own}]}
        response = {'request_id': 'q', 'status': 'completed', 'results': [own, target]}
        self.assertEqual(m.evaluate(gt, [response], k=1)['overall']['recall_at_k'], 0)
        self.assertEqual(m.evaluate(gt, [response], k=2)['overall']['recall_at_k'], 1)
    def test_one_prediction_cannot_count_two_targets(self):
        targets = [self.p([0, 0, 10, 10]), self.p([3, 0, 13, 10])]
        self.assertEqual(len(m.match_targets([self.p([1, 0, 12, 10])], targets, .5)), 1)
    def test_bipartite_assignment_handles_ambiguous_boxes(self):
        targets = [self.p([0, 0, 10, 10]), self.p([3, 0, 13, 10])]
        predictions = [self.p([1, 0, 12, 10]), self.p([0, 0, 7, 10])]
        self.assertEqual(len(m.match_targets(predictions, targets, .5)), 2)
    def test_missing_and_incomplete_remain_in_denominator(self):
        gt = {'queries': [{'request_id': 'a', 'targets': [self.p([0,0,10,10])]}, {'request_id': 'b', 'targets': [self.p([0,0,10,10])]}]}
        out = m.evaluate(gt, [{'request_id': 'a', 'status': 'incomplete', 'results': []}])
        self.assertEqual(out['overall']['positive_pairs'], 2)
        self.assertEqual(out['overall']['incomplete_or_missing'], 2)
        self.assertIsNone(out['overall']['coarse_candidates_recall'])
    def test_whole_image_not_automatically_good_coarse_recall(self):
        target = self.p([0,0,10,10]);gt = {'queries': [{'request_id':'q', 'targets':[target]}]}
        out = m.evaluate(gt, [{'request_id':'q','status':'completed','results':[], 'coarse_candidates':[self.p([0,0,1000,1000])]}])
        self.assertEqual(out['overall']['coarse_candidates_recall'], 0)
    def test_duplicate_response_rejected(self):
        r = {'request_id': 'q', 'status': 'completed', 'results': []}
        with self.assertRaises(ValueError): m.evaluate({'queries': []}, [r, r])
    def test_negative_query_precision_not_invented(self):
        out = m.evaluate({'queries':[{'request_id':'n','targets':[]}]}, [{'request_id':'n','status':'completed','results':[self.p([0,0,1,1])]}])
        self.assertIsNone(out['queries'][0]['precision_at_returned_k'])
        self.assertIsNone(out['overall']['recall_at_k'])

if __name__ == '__main__': unittest.main()

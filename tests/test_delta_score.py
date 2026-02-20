from collections import namedtuple
import os
import unittest

from spliceai_mlx.utils import Annotator, get_delta_scores

Record = namedtuple('Record', ['chrom', 'pos', 'ref', 'alts'])

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class TestDeltaScore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ann = Annotator(
            os.path.join(_DATA_DIR, 'test.fa'), 'grch37')
        cls.ann_without_prefix = Annotator(
            os.path.join(_DATA_DIR, 'test_without_prefix.fa'), 'grch37')

    def test_get_delta_score_acceptor(self):
        record = Record('10', 94077, 'A', ['C'])
        scores = get_delta_scores(record, self.ann, 500, 0)
        self.assertEqual(scores, ['C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193'])
        scores = get_delta_scores(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ['C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193'])

        record = Record('chr10', 94077, 'A', ['C'])
        scores = get_delta_scores(record, self.ann, 500, 0)
        self.assertEqual(scores, ['C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193'])
        scores = get_delta_scores(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ['C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193'])

    def test_get_delta_score_donor(self):
        record = Record('10', 94555, 'C', ['T'])
        scores = get_delta_scores(record, self.ann, 500, 0)
        self.assertEqual(scores, ['T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0'])
        scores = get_delta_scores(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ['T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0'])

        record = Record('chr10', 94555, 'C', ['T'])
        scores = get_delta_scores(record, self.ann, 500, 0)
        self.assertEqual(scores, ['T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0'])
        scores = get_delta_scores(record, self.ann_without_prefix, 500, 0)
        self.assertEqual(scores, ['T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0'])

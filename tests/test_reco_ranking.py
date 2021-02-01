# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pandas as pd

from jurity.recommenders import RankingRecoMetrics
from jurity.recommenders.ndcg import idcg
from jurity.utils import Constants


class TestRankingRecommenders(unittest.TestCase):

    def test_ndcg(self):
        # First, test the IDCG value
        idcg_val = idcg(3)
        self.assertEqual(1. / np.log2(2) + 1. / np.log2(3) + 1. / np.log2(4), idcg_val)

        # Then, test NDCG
        # User 1 has items 1, 4, 2 relevant and was recommended items 1, 3, 2
        # User 2 checks for no relevant items, shouldn't contribute to the metric
        # User 3 checks for no recommendations, is 0
        # User 4 has items 1, 2 relevant and was recommended items 3, 4; is 0
        actual = pd.DataFrame({Constants.user_id: [1, 1, 1, 1, 3, 4, 4],
                               Constants.item_id: [1, 2, 3, 4, 3, 1, 2],
                               'click': [1, 1, 0, 1, 1, 1, 1]})

        predicted = pd.DataFrame({Constants.user_id: [1, 1, 1, 2, 4, 4],
                                  Constants.item_id: [1, 2, 3, 3, 3, 4],
                                  'click': [0.8, 0.7, 0.75, 0.7, 0.6, 0.4]})

        metric = RankingRecoMetrics.NDCG(click_column='click', k=3)
        results = metric.get_score(actual, predicted, return_extended_results=True)
        self.assertEqual(((1. / np.log2(2) + 1. / np.log2(4)) / idcg_val) / 3, results['ndcg'])
        self.assertEqual(3, results['support'])

    def test_precision(self):
        # User 1 was recommended items 1, 3, 2 and has items 1, 4 relevant
        # User 2 and 3 check for no relevant items
        # User 4 checks for no recommendations
        actual = pd.DataFrame({Constants.user_id: [1, 1, 1, 3, 4],
                               Constants.item_id: [1, 2, 4, 1, 3],
                               'click': [1, 0, 1, 0, 1]})

        predicted = pd.DataFrame({Constants.user_id: [1, 1, 1, 2, 3],
                                  Constants.item_id: [1, 2, 3, 3, 1],
                                  'click': [0.8, 0.7, 0.75, 0.7, 0.5]})

        metric = RankingRecoMetrics.Precision(click_column='click', k=2)
        results = metric.get_score(actual, predicted, return_extended_results=True)

        self.assertEqual(0.5, results['precision'])
        self.assertEqual(1, results['support'])

        precision_3 = RankingRecoMetrics.Precision(click_column='click', k=3)
        result_3 = precision_3.get_score(actual, predicted)
        self.assertEqual(1. / 3, result_3)

    def test_precision_at_max_recs(self):
        """Tests Precision@k for the case when all users have exactly k recommendations.

        When all users have exactly k recommendations,
        there isn't an extra ``user_id`` index generated when sorting for the largest ``k`` scores.
        """
        actual = pd.DataFrame({Constants.user_id: [0], Constants.item_id: [0], 'click': [True]})
        predicted = pd.DataFrame(
            {Constants.user_id: [0, 0, 0], Constants.item_id: [0, 1, 2], 'click': [0, -1, -2]})

        self.assertEqual(1., RankingRecoMetrics.Precision('click', k=1).get_score(actual, predicted))
        self.assertEqual(0.5, RankingRecoMetrics.Precision('click', k=2).get_score(actual, predicted))
        self.assertEqual(1. / 3, RankingRecoMetrics.Precision('click', k=3).get_score(actual, predicted))

    def test_map(self):
        # User 1 got items 1,3,2,4 as recommendations. Items 1 and 4 are relevant.
        # User 2 checks for no relevant items
        # user 3 checks for no recommendations
        actual = pd.DataFrame({Constants.user_id: [1, 1, 1, 3],
                               Constants.item_id: [1, 2, 4, 3],
                               'click': [1, 0, 1, 1]})

        predicted = pd.DataFrame({Constants.user_id: [1, 1, 1, 1, 2],
                                  Constants.item_id: [1, 2, 3, 4, 3],
                                  'click': [0.8, 0.7, 0.75, 0.65, 0.7]})

        metric = RankingRecoMetrics.MAP(click_column='click', k=2)
        results = metric.get_score(actual, predicted, return_extended_results=True)

        self.assertEqual(0.5, results['map'])
        self.assertEqual(1, results['support'])

        map_3 = RankingRecoMetrics.MAP(click_column='click', k=3).get_score(actual, predicted)
        self.assertEqual(0.5, map_3)

        map_4 = RankingRecoMetrics.MAP(click_column='click', k=4).get_score(actual, predicted)
        self.assertEqual(0.75, map_4)

    def test_recall(self):
        # User 1 has items 1, 4, 2 relevant and was recommended items 1, 3, 2, so they should be included in the support
        # User 2 checks for no relevant items, so they shouldn't be included in the support
        # User 3 & 4 checks for no recommendations, but they should be included in the support
        actual = pd.DataFrame({Constants.user_id: [1, 1, 1, 1, 3, 4],
                               Constants.item_id: [1, 2, 3, 4, 3, 1],
                               'click': [1, 1, 0, 1, 1, 1]})

        predicted = pd.DataFrame({Constants.user_id: [1, 1, 1, 2],
                                  Constants.item_id: [1, 2, 3, 3],
                                  'click': [0.8, 0.7, 0.75, 0.7]})

        metric = RankingRecoMetrics.Recall(click_column='click', k=2)
        results = metric.get_score(actual, predicted, return_extended_results=True)

        self.assertEqual(1. / 9, results['recall'])
        self.assertEqual(3, results['support'])

        recall_3 = RankingRecoMetrics.Recall(click_column='click', k=3).get_score(actual, predicted)
        self.assertEqual(2. / 9, recall_3)

    def test_recall_at_max_recs(self):
        """Tests Recall@k for the case when all users have exactly k recommendations.

        When all users have exactly k recommendations, there isn't an extra ``user_id`` index generated when sorting
        for the largest ``k`` scores.
        """

        actual = pd.DataFrame({Constants.user_id: [0, 0],
                               Constants.item_id: [1, 2],
                               'click': [True, True]})

        predicted = pd.DataFrame({Constants.user_id: [0, 0, 0, 0],
                                  Constants.item_id: [0, 1, 2, 3],
                                  'click': [0, -1, -2, -3]})

        self.assertEqual(0., RankingRecoMetrics.Recall('click', k=1).get_score(actual, predicted))
        self.assertEqual(0.5, RankingRecoMetrics.Recall('click', k=2).get_score(actual, predicted))
        self.assertEqual(1., RankingRecoMetrics.Recall('click', k=3).get_score(actual, predicted))
        self.assertEqual(1., RankingRecoMetrics.Recall('click', k=4).get_score(actual, predicted))

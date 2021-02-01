# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import pandas as pd

from jurity.recommenders import BinaryRecoMetrics, CombinedMetrics, RankingRecoMetrics
from jurity.utils import Constants


class TestCombinedRecommenders(unittest.TestCase):

    def test_change_column_names(self):
        user_id_column = 'uid'
        item_id_column = 'iid'
        actual = pd.DataFrame({user_id_column: [0, 0], item_id_column: [1, 2], 'click': [True, True]})
        predicted = pd.DataFrame({user_id_column: [0, 0, 0, 0], item_id_column: [0, 1, 2, 3], 'click': [0, -1, -2, -3]})

        # Test that the output is the same
        recall_1 = RankingRecoMetrics.Recall('click', k=1, user_id_column=user_id_column, item_id_column=item_id_column)
        self.assertEqual(0., recall_1.get_score(actual, predicted))

        recall_2 = RankingRecoMetrics.Recall('click', k=2, user_id_column=user_id_column, item_id_column=item_id_column)
        self.assertEqual(0.5, recall_2.get_score(actual, predicted))

        # Test that none of the metrics crash
        metrics = CombinedMetrics(
            BinaryRecoMetrics.CTR('click', k=1, user_id_column=user_id_column, item_id_column=item_id_column),
            RankingRecoMetrics.MAP('click', k=1, user_id_column=user_id_column, item_id_column=item_id_column),
            RankingRecoMetrics.NDCG('click', k=1, user_id_column=user_id_column, item_id_column=item_id_column),
            RankingRecoMetrics.Precision('click', k=1, user_id_column=user_id_column, item_id_column=item_id_column),
            RankingRecoMetrics.Recall('click', k=1, user_id_column=user_id_column, item_id_column=item_id_column))

        metrics.get_score(actual, predicted)

    def test_accumulate_when_no_results_this_batch(self):
        metrics = CombinedMetrics(RankingRecoMetrics.Recall(click_column='click', k=1),
                                  BinaryRecoMetrics.CTR(click_column='click', k=1))

        actual = pd.DataFrame({Constants.user_id: [0], Constants.item_id: [0], 'click': [True]})
        predicted = pd.DataFrame({Constants.user_id: [0], Constants.item_id: [0], 'click': [0]})
        batch_res, acc_res = metrics.get_score(actual, predicted, batch_accumulate=True, return_extended_results=True)
        self.assertEqual(2, len(batch_res))
        self.assertEqual(2, len(acc_res))

        actual = pd.DataFrame({Constants.user_id: [1], Constants.item_id: [1], 'click': [True]})
        predicted = pd.DataFrame({Constants.user_id: [2], Constants.item_id: [2], 'click': [0]})
        batch_res, acc_res = metrics.get_score(actual, predicted, batch_accumulate=True, return_extended_results=True)
        self.assertEqual(2, len(batch_res))
        self.assertEqual(2, len(acc_res))

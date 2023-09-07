# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import pandas as pd

from jurity.fairness import BinaryFairnessMetrics, MultiClassFairnessMetrics
from jurity.mitigation import BinaryMitigation
from jurity.recommenders import BinaryRecoMetrics, RankingRecoMetrics


class TestJurity(unittest.TestCase):

    def test_fairness_quick_start_example(self):
        # Data
        binary_predictions = [1, 1, 0, 1, 0, 0]
        multi_class_predictions = ["a", "b", "c", "b", "a", "a"]
        multi_class_multi_label_predictions = [["a", "b"], ["b", "c"], ["b"], ["a", "b"], ["c", "a"], ["c"]]
        is_member = [0, 0, 0, 1, 1, 1]
        classes = ["a", "b", "c"]

        # Metric (see also other available metrics)
        metric = BinaryFairnessMetrics.StatisticalParity()
        multi_metric = MultiClassFairnessMetrics.StatisticalParity(list_of_classes=classes)

        # Score
        binary_score = metric.get_score(binary_predictions, is_member)
        multi_scores = multi_metric.get_scores(multi_class_predictions, is_member)
        multi_label_scores = multi_metric.get_scores(multi_class_multi_label_predictions, is_member)

        # Results
        self.assertEqual(metric.description, "Measures the difference in statistical parity between two groups")
        self.assertEqual(metric.lower_bound, -0.2)
        self.assertEqual(metric.upper_bound, 0.2)
        self.assertEqual(metric.ideal_value, 0)
        self.assertEqual(binary_score, -0.3333333333333333)
        self.assertListEqual(multi_scores, [0.3333333333333333, 0.0, -0.3333333333333333])
        self.assertListEqual(multi_label_scores, [0.3333333333333333, -0.6666666666666667, 0.3333333333333333])

    def test_mitigation_quick_start_example(self):
        # Data
        labels = [1, 1, 0, 1, 0, 0, 1, 0]
        predictions = [0, 0, 0, 1, 1, 1, 1, 0]
        likelihoods = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1]
        is_member = [0, 0, 0, 0, 1, 1, 1, 1]

        # Bias Mitigation
        mitigation = BinaryMitigation.EqualizedOdds()

        # Training: Learn mixing rates from the labeled data
        mitigation.fit(labels, predictions, likelihoods, is_member)

        # Testing: Mitigate bias in predictions
        fair_predictions, fair_likelihoods = mitigation.transform(predictions, likelihoods, is_member)

        # Results: Fairness before and after
        before_scores = BinaryFairnessMetrics().get_all_scores(labels, predictions, is_member)
        after_scores = BinaryFairnessMetrics().get_all_scores(labels, fair_predictions, is_member)

        before_scores_check = {'Average Odds': 0.667,
                               'Disparate Impact': 3.0,
                               'Equal Opportunity': 0.667,
                               'FNR difference': -0.667,
                               'FOR difference': -0.667,
                               'Generalized Entropy Index': 0.25,
                               'Predictive Equality': 0.667,
                               'Statistical Parity': 0.5,
                               'Theil Index': 0.347}

        after_scores_check = {'Average Odds': 0.0,
                              'Disparate Impact': 1.0,
                              'Equal Opportunity': 0.333,
                              'FNR difference': -0.333,
                              'FOR difference': -1.0,
                              'Generalized Entropy Index': 0.14,
                              'Predictive Equality': -0.333,
                              'Statistical Parity': 0.0,
                              'Theil Index': 0.193}

        self.assertDictEqual(before_scores["Value"].to_dict(), before_scores_check)
        self.assertDictEqual(after_scores["Value"].to_dict(), after_scores_check)

    def test_reco_quick_start_example(self):
        # Data
        actual = pd.DataFrame({"user_id": [1, 2, 3, 4], "item_id": [1, 2, 0, 3], "clicks": [0, 1, 0, 0]})
        predicted = pd.DataFrame({"user_id": [1, 2, 3, 4], "item_id": [1, 2, 2, 3], "clicks": [0.8, 0.7, 0.8, 0.7]})

        # Metrics
        auc = BinaryRecoMetrics.AUC(click_column="clicks")
        ctr = BinaryRecoMetrics.CTR(click_column="clicks")
        ncdg_k = RankingRecoMetrics.NDCG(click_column="clicks", k=3)
        precision_k = RankingRecoMetrics.Precision(click_column="clicks", k=2)
        recall_k = RankingRecoMetrics.Recall(click_column="clicks", k=2)
        map_k = RankingRecoMetrics.MAP(click_column="clicks", k=2)

        # Scores
        self.assertEqual(auc.get_score(actual, predicted), 0.25)
        self.assertEqual(ctr.get_score(actual, predicted), 0.3333333333333333)
        self.assertEqual(ncdg_k.get_score(actual, predicted), 1)
        self.assertEqual(precision_k.get_score(actual, predicted), 1)
        self.assertEqual(recall_k.get_score(actual, predicted), 1)
        self.assertEqual(map_k.get_score(actual, predicted), 1)

# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pandas as pd

from jurity.fairness import BinaryFairnessMetrics
from jurity.utils import InputShapeError


class TestBinaryFairness(unittest.TestCase):

    def test_stat_parity_normal_np(self):

        # Data
        y_pred = np.array([1, 1, 0, 1, 0, 0])
        is_member = np.array([0, 0, 0, 1, 1, 1])

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        score = metric.get_score(y_pred, is_member)

        assert np.isclose(score, -0.3333, atol=0.001)

    def test_stat_parity_normal_df(self):

        # Data
        my_df = pd.DataFrame.from_dict({'y_pred': [1, 1, 0, 1, 0, 0],
                                        'is_member': [0, 0, 0, 1, 1, 1]})

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        score = metric.get_score(my_df['y_pred'], my_df['is_member'])

        assert np.isclose(score, -0.3333, atol=0.001)

    def test_stat_parity_normal_list(self):

        # Data
        y_pred = [1, 1, 0, 1, 0, 0]
        is_member = [0, 0, 0, 1, 1, 1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        score = metric.get_score(y_pred, is_member)

        assert np.isclose(score, -0.3333, atol=0.001)

    def test_stat_parity_invalid_list(self):

        # Data
        y_pred = [1, 1, 0, 1, 0]
        is_member = [0, 0, 0, 1, 1, 1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        with self.assertRaises(InputShapeError):
            metric.get_score(y_pred, is_member)

    def test_stat_parity_invalid_df(self):

        # Data
        my_df = pd.DataFrame.from_dict({'y_pred': [1, 1, 0, 1, 0, 2],
                                        'is_member': [0, 0, 0, 1, 1, 1]})

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(my_df['y_pred'], my_df['is_member'])

    def test_stat_parity_invalid_np(self):

        # Data
        y_pred = np.array([1, 1, 0, 1, 0])
        is_member = np.array([0, 0, 0, 1, 1, 1])

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        with self.assertRaises(InputShapeError):
            metric.get_score(y_pred, is_member)

    def test_stat_parity_edge_1(self):

        # Data: edge cases stat parity == 1
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        stat_parity = metric.get_score(y_pred, is_member)

        assert stat_parity == 1

    def test_stat_parity_edge_2(self):

        # Data: edge case stat parity == -1
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        is_member = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()
        score = metric.get_score(y_pred, is_member)

        assert score == -1

    def test_avg_odds_diff_edge_1(self):

        # Data: edge case - homogeneous ground truth within group - returns None
        # unprivileged homogeneous
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        is_member = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Metric
        metric = BinaryFairnessMetrics.AverageOdds()

        # Score
        with self.assertWarns(UserWarning):
            assert metric.get_score(y_true, y_pred, is_member) is None

    def test_avg_odds_diff_edge_2(self):

        # Data
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        is_member = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # privileged homogeneous
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])

        # Metric
        metric = BinaryFairnessMetrics.AverageOdds()

        # Score
        with self.assertWarns(UserWarning):
            assert metric.get_score(y_true, y_pred, is_member) is None

    def test_avg_odds_diff_edge_3(self):

        # Data
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        is_member = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # both homogeneous
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Metric
        metric = BinaryFairnessMetrics.AverageOdds()

        # Score
        with self.assertWarns(UserWarning):
            assert metric.get_score(y_true, y_pred, is_member) is None

    def test_avg_odds_diff_edge_4(self):

        # Data: edge case of 1
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.AverageOdds()

        # Score
        assert metric.get_score(y_true, y_pred, is_member) == 1

    def test_avg_odds_diff_edge_5(self):

        # Data
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        # edge case of - 1
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Metric
        metric = BinaryFairnessMetrics.AverageOdds()

        # Score
        assert metric.get_score(y_true, y_pred, is_member) == -1

    def test_avg_odds_diff_normal_np(self):

        # Data: medium number
        y_true = np.array([0, 1, 0, 1, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.AverageOdds()

        # Score
        score = metric.get_score(y_true, y_pred, is_member)

        assert np.isclose(score, -0.5833, atol=0.01)

    def test_avg_odds_diff_normal_list(self):

        # Data: medium number
        y_true = [0, 1, 0, 1, 1, 1, 1, 0]
        y_pred = [0, 0, 1, 0, 0, 1, 1, 1]
        is_member = [1, 1, 1, 1, 0, 0, 0, 0]

        # Metric
        metric = BinaryFairnessMetrics.AverageOdds()

        # Score
        score = metric.get_score(y_true, y_pred, is_member)

        assert np.isclose(score, -0.5833, atol=0.01)

    def test_avg_odds_diff_normal_df(self):
        # Data: medium number
        my_df = pd.DataFrame.from_dict({'y_true': [0, 1, 0, 1, 1, 1, 1, 0],
                                        'y_pred': [0, 0, 1, 0, 0, 1, 1, 1],
                                        'is_member': [1, 1, 1, 1, 0, 0, 0, 0]})

        # Metric
        metric = BinaryFairnessMetrics.AverageOdds()

        # Score
        score = metric.get_score(my_df['y_true'], my_df['y_pred'], my_df['is_member'])

        assert np.isclose(score, -0.5833, atol=0.01)

    def test_avg_odds_diff_normal_invalid(self):
        # Data: medium number
        my_df = pd.DataFrame.from_dict({'y_true': [0, 1, 0, 1, 1, 1, 1, 0],
                                        'y_pred': [0, 0, 1, 0, 0, 1, 1, -1],
                                        'is_member': [1, 1, 1, 1, 0, 0, 0, 0]})

        # Metric
        metric = BinaryFairnessMetrics.AverageOdds()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(my_df['y_true'], my_df['y_pred'], my_df['is_member'])

    def test_pred_equality_edge_1(self):

        # Data: edge case - homogeneous ground truth within group - returns None
        # unprivileged homogeneous
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.PredictiveEquality()

        with self.assertWarns(UserWarning):
            assert metric.get_score(y_true, y_pred, is_member) is None

    def test_pred_equality_edge_2(self):

        # Data: edge case - homogeneous ground truth within group - returns None
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        # privileged homogeneous
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])

        # Metric
        pred_eq_fun = BinaryFairnessMetrics.PredictiveEquality()

        # Score
        with self.assertWarns(UserWarning):
            assert pred_eq_fun.get_score(y_true, y_pred, is_member) is None

    def test_pred_equality_edge_3(self):

        # Data: edge case - homogeneous ground truth within group - returns None
        # both homogeneous
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.PredictiveEquality()

        # Score
        with self.assertWarns(UserWarning):
            assert metric.get_score(y_true, y_pred, is_member) is None

    def test_pred_equality_edge_4(self):

        # Data: edge case - homogeneous ground truth within group - returns None
        # edge case of 1
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.PredictiveEquality()

        # Score
        assert metric.get_score(y_true, y_pred, is_member) == 1

    def test_pred_equality_edge_5(self):

        # edge case of - 1
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        metric = BinaryFairnessMetrics.PredictiveEquality()

        assert metric.get_score(y_true, y_pred, is_member) == -1

    def test_pred_equality_normal_invalid(self):

        # Data: edge case - homogeneous ground truth within group - returns None
        # medium number
        y_true = np.array([0, 1, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.PredictiveEquality()

        # Score
        with self.assertRaises(InputShapeError):
            metric.get_score(y_true, y_pred, is_member)

    def test_pred_equality_normal_np(self):

        # Data: edge case - homogeneous ground truth within group - returns None
        # medium number
        y_true = np.array([0, 1, 0, 1, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.PredictiveEquality()

        # Score
        score = metric.get_score(y_true, y_pred, is_member)

        assert np.isclose(score, -0.5, atol=0.001)

    def test_pred_equality_normal_df(self):

        # medium number
        my_df = pd.DataFrame.from_dict({'y_true': [0, 1, 0, 1, 1, 1, 1, 0],
                                        'y_pred': [0, 0, 1, 0, 0, 1, 1, 1],
                                        'is_member': [1, 1, 1, 1, 0, 0, 0, 0]})

        # Metric
        metric = BinaryFairnessMetrics.PredictiveEquality()

        # Score
        score = metric.get_score(my_df['y_true'], my_df['y_pred'], my_df['is_member'])

        assert np.isclose(score, -0.5, atol=0.001)

    def test_pred_equality_normal_list(self):

        # Data: edge case - homogeneous ground truth within group - returns None
        # medium number
        y_true = [0, 1, 0, 1, 1, 1, 1, 0]
        y_pred = [0, 0, 1, 0, 0, 1, 1, 1]
        is_member = [1, 1, 1, 1, 0, 0, 0, 0]

        # Metric
        metric = BinaryFairnessMetrics.PredictiveEquality()

        # Score
        score = metric.get_score(y_true, y_pred, is_member)

        assert np.isclose(score, -0.5, atol=0.001)

    def test_equal_opp_normal_invalid(self):

        # Data
        y_true = np.array([1, 0, 0, 0, 1, 1, 0, 2])
        y_pred = np.array([0, 1, 1, 1, 1, 1, 1, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.EqualOpportunity()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(y_true, y_pred, is_member)

    def test_equal_opp_normal_np(self):

        # Data
        y_true = np.array([1, 0, 0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 1, 1, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.EqualOpportunity()

        # Score
        score = metric.get_score(y_true, y_pred, is_member)

        assert np.isclose(score, -0.666, atol=0.001)

    def test_equal_opp_normal_list(self):

        # Data
        y_true = [1, 0, 0, 0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 1, 1, 1, 0]
        is_member = [1, 1, 1, 1, 0, 0, 0, 0]

        # Metric
        metric = BinaryFairnessMetrics.EqualOpportunity()

        # Score
        score = metric.get_score(y_true, y_pred, is_member)

        assert np.isclose(score, -0.666, atol=0.001)

    def test_equal_opp_normal_df(self):

        # medium number
        my_df = pd.DataFrame.from_dict({'y_true': [1, 0, 0, 0, 1, 1, 0, 1],
                                        'y_pred': [0, 1, 1, 1, 1, 1, 1, 0],
                                        'is_member': [1, 1, 1, 1, 0, 0, 0, 0]})

        # Metric
        metric = BinaryFairnessMetrics.EqualOpportunity()

        # Score
        score = metric.get_score(my_df['y_true'], my_df['y_pred'], my_df['is_member'])

        assert np.isclose(score, -0.666, atol=0.001)

    def test_equal_opp_edge_1(self):

        # Data: edge cases equal opp == 1
        y_true = np.array([1, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.EqualOpportunity()

        # Score
        with self.assertWarns(UserWarning):  # division by zero caught inside numpy
            assert metric.get_score(y_true, y_pred, is_member) == 1

    def test_equal_opp_edge_2(self):

        # Data
        y_true = np.array([1, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        # edge case equal opp == -1
        is_member = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Metric
        metric = BinaryFairnessMetrics.EqualOpportunity()

        with self.assertWarns(UserWarning):  # division by zero caught inside numpy
            assert metric.get_score(y_true, y_pred, is_member) == -1

    def test_equal_opp_edge_3(self):

        # Data: homogeneous both groups in ground truth - returns nan
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 1, 1, 0])
        is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        # Metric
        metric = BinaryFairnessMetrics.EqualOpportunity()

        with self.assertWarns(UserWarning):  # division by zero caught inside numpy
            metric.get_score(y_true, y_pred, is_member)

    def test_disp_impact_edge1(self):

        # Metric
        metric = BinaryFairnessMetrics.DisparateImpact()

        # Data
        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        # test no positives in protected
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
        assert metric.get_score(y_pred, is_member) == 0

    def test_disp_impact_edge2(self):

        # Metric
        metric = BinaryFairnessMetrics.DisparateImpact()

        # Data
        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        # test no positives in unprotected
        y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

        with self.assertWarns(UserWarning):
            assert metric.get_score(y_pred, is_member) == 1

    def test_disp_impact_edge3(self):

        # Metric
        metric = BinaryFairnessMetrics.DisparateImpact()

        # Data
        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        # test 1
        y_pred = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        assert metric.get_score(y_pred, is_member) == 1

    def test_disp_impact_normal_np(self):

        # Metric
        metric = BinaryFairnessMetrics.DisparateImpact()

        # Data
        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        # test a medium number
        y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        assert metric.get_score(y_pred, is_member) == 2

    def test_disp_impact_normal_list(self):

        # Metric
        metric = BinaryFairnessMetrics.DisparateImpact()

        # Data
        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        # test a medium number
        y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        assert metric.get_score(y_pred, is_member) == 2

    def test_disp_impact_normal_df(self):

        # Metric
        metric = BinaryFairnessMetrics.DisparateImpact()

        # medium number
        my_df = pd.DataFrame.from_dict({'y_pred': [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                        'is_member': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]})

        # Score
        assert metric.get_score(my_df['y_pred'], my_df['is_member']) == 2

    def test_disp_impact_normal_invalid(self):

        # Metric
        metric = BinaryFairnessMetrics.DisparateImpact()

        # medium number
        my_df = pd.DataFrame.from_dict({'y_pred': ['0', '0', '0', 1, 1, 1, '0', '0', '0', '0'],
                                        'is_member': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]})

        # Score
        with self.assertRaises(TypeError):
            metric.get_score(my_df['y_pred'], my_df['is_member'])

    def test_fnr_diff_normal_invalid(self):

        # Metric
        metric = BinaryFairnessMetrics.FNRDifference()

        # Data
        y_true = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 2])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0])

        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            metric.get_score(y_true, y_pred, is_member)

    def test_fnr_diff_normal_np(self):

        # Metric
        metric = BinaryFairnessMetrics.FNRDifference()

        # Data
        y_true = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0])

        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        assert np.isclose(metric.get_score(y_true, y_pred, is_member), 0.333, atol=0.001)

    def test_fnr_diff_normal_list(self):

        # Metric
        metric = BinaryFairnessMetrics.FNRDifference()

        # Data
        y_true = [0, 1, 1, 0, 1, 1, 1, 0, 1, 0]
        y_pred = [0, 0, 1, 0, 0, 1, 1, 1, 0, 0]

        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        assert np.isclose(metric.get_score(y_true, y_pred, is_member), 0.333, atol=0.001)

    def test_fnr_diff_normal_df(self):

        my_df = pd.DataFrame.from_dict({'y_true': [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
                                        'y_pred': [0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
                                        'is_member': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]})
        # Metric
        metric = BinaryFairnessMetrics.FNRDifference()

        # Score
        score = metric.get_score(my_df['y_true'], my_df['y_pred'], my_df['is_member'])

        assert np.isclose(score, 0.333, atol=0.001)

    def test_fnr_diff_edge1(self):

        # Metric
        metric = BinaryFairnessMetrics.FNRDifference()

        # edge case of 1
        y_true = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        assert metric.get_score(y_true, y_pred, is_member) == 1

    def test_fnr_diff_edge2(self):

        # Metric
        metric = BinaryFairnessMetrics.FNRDifference()

        # edge case of -1
        y_true = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        is_member = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        assert metric.get_score(y_true, y_pred, is_member) == -1

    def test_gei_normal_np(self):

        # Metric
        metric = BinaryFairnessMetrics.GeneralizedEntropyIndex()

        y_true = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        score = metric.get_score(y_true, y_pred)
        assert isinstance(score, float)
        assert np.isclose(score, 0.302, atol=0.01)

    def test_gei_normal_invalid(self):

        # Metric
        metric = BinaryFairnessMetrics.GeneralizedEntropyIndex()

        y_true = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, -1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        with self.assertRaises(ValueError):
            metric.get_score(y_true, y_pred)

    def test_gei_normal_df(self):

        my_df = pd.DataFrame.from_dict({'y_true': [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
                                        'y_pred': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})

        # Metric
        metric = BinaryFairnessMetrics.GeneralizedEntropyIndex()

        # Score
        score = metric.get_score(my_df['y_true'], my_df['y_pred'])

        assert isinstance(score, float)
        assert np.isclose(score, 0.302, atol=0.01)

    def test_gei_normal_list(self):

        # Metric
        metric = BinaryFairnessMetrics.GeneralizedEntropyIndex()

        y_true = [0, 1, 1, 0, 1, 1, 1, 0, 1, 0]
        y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        score = metric.get_score(y_true, y_pred)
        assert isinstance(score, float)
        assert np.isclose(score, 0.302, atol=0.01)

    def test_gei_alpha_zero(self):

        np.random.seed(1)

        # test bounds
        metric = BinaryFairnessMetrics.GeneralizedEntropyIndex()
        my_dict = {}
        for _ in range(1000):
            random = np.random.choice([0, 1], 10)
            if str(random) in my_dict:
                continue
            else:
                my_dict[str(random)] = (random[:5], random[5:])

        with self.assertWarns(RuntimeWarning):  # division by 0 in certain instances.
            # enumerate through all the combinations to find the max
            # alpha = 0
            alpha = 0
            results = []
            vals = []

            for y_true, y_pred in my_dict.values():
                vals.append([y_true, y_pred])
                results.append(metric.get_score(y_true, y_pred, alpha=alpha))

            assert min(results) == 0
            assert max(results) == np.inf

    def test_gei_alpha_one(self):

        np.random.seed(1)

        # test bounds
        metric = BinaryFairnessMetrics.GeneralizedEntropyIndex()
        my_dict = {}
        for _ in range(1000):
            random = np.random.choice([0, 1], 10)
            if str(random) in my_dict:
                continue
            else:
                my_dict[str(random)] = (random[:5], random[5:])

        with self.assertWarns(RuntimeWarning):  # division by 0 in certain instances.
            # enumerate through all the combinations
            # alpha = 1
            # test theil as well
            alpha = 1
            results = []
            vals = []

            for y_true, y_pred in my_dict.values():
                vals.append([y_true, y_pred])
                results.append(metric.get_score(y_true, y_pred, alpha=alpha))

            assert min(results) == 0
            assert max(results) == np.log(5)

    def test_gei_alpha_more_than_one(self):

        np.random.seed(1)

        # test bounds
        metric = BinaryFairnessMetrics.GeneralizedEntropyIndex()
        my_dict = {}
        for _ in range(1000):
            random = np.random.choice([0, 1], 10)
            if str(random) in my_dict:
                continue
            else:
                my_dict[str(random)] = (random[:5], random[5:])

        with self.assertWarns(RuntimeWarning):  # division by 0 in certain instances.
            # enumerate through all the combinations
            alpha = 3
            results = []
            vals = []
            for y_true, y_pred in my_dict.values():
                vals.append([y_true, y_pred])
                results.append(metric.get_score(y_true, y_pred, alpha=alpha))

            assert min(results) == 0
            assert max(results) == (np.power(5, alpha - 1) - 1) / (alpha * (alpha - 1))

    def test_theil_normal_np(self):

        # Metric
        metric = BinaryFairnessMetrics.TheilIndex()

        y_true = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        score = metric.get_score(y_true, y_pred)
        assert isinstance(score, float)
        assert np.isclose(score, 0.413, atol=0.01)

    def test_theil_normal_df(self):

        my_df = pd.DataFrame.from_dict({'y_true': [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
                                        'y_pred': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})

        # Metric
        metric = BinaryFairnessMetrics.TheilIndex()
        # Score
        score = metric.get_score(my_df['y_true'], my_df['y_pred'])

        assert isinstance(score, float)
        assert np.isclose(score, 0.413, atol=0.01)

    def test_theil_normal_list(self):

        # Metric
        metric = BinaryFairnessMetrics.TheilIndex()

        y_true = [0, 1, 1, 0, 1, 1, 1, 0, 1, 0]
        y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        score = metric.get_score(y_true, y_pred)
        assert isinstance(score, float)
        assert np.isclose(score, 0.413, atol=0.01)

    def test_theil_normal_invalid(self):

        # Metric
        metric = BinaryFairnessMetrics.TheilIndex()

        y_true = [0, 1, 1, 0, 1, 1, 1, 0, 1, -1]
        y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        with self.assertRaises(ValueError):
            metric.get_score(y_true, y_pred)

    def test_theil_boundaries(self):
        np.random.seed(1)

        # test bounds
        metric = BinaryFairnessMetrics.TheilIndex()
        my_dict = {}
        for _ in range(1000):
            random = np.random.choice([0, 1], 10)
            if str(random) in my_dict:
                continue
            else:
                my_dict[str(random)] = (random[:5], random[5:])

        with self.assertWarns(RuntimeWarning):  # division by 0 in certain instances.
            # enumerate through all the combinations
            # test theil as well
            results_thiel = []
            vals = []

            for y_true, y_pred in my_dict.values():
                vals.append([y_true, y_pred])
                results_thiel.append(metric.get_score(y_true, y_pred))

            assert min(results_thiel) == 0
            assert max(results_thiel) == np.log(5)

    @staticmethod
    def extract_metric_from_df(metric_name, df, attribute='Value'):
        return df.loc[df.index == metric_name, attribute].values.squeeze()

    def test_all_scores_valid(self):

        # test standard pandas table creation
        y_true = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        df = BinaryFairnessMetrics.get_all_scores(y_true, y_pred, is_member)
        assert type(df) == pd.DataFrame

        assert self.extract_metric_from_df('Statistical Parity', df) == 0.
        assert self.extract_metric_from_df('Average Odds', df) == 0.375
        assert self.extract_metric_from_df('Disparate Impact', df) == 1.
        assert self.extract_metric_from_df('FNR difference', df) == 0.25
        assert self.extract_metric_from_df('Predictive Equality', df) == 1.00
        assert self.extract_metric_from_df('Generalized Entropy Index', df) == 1.375
        assert np.isclose(self.extract_metric_from_df('Theil Index', df), 1.263, atol=0.01)

        attr = 'Ideal Value'
        assert self.extract_metric_from_df('Average Odds', df, attr) == 0
        assert self.extract_metric_from_df('Disparate Impact', df, attr) == 1
        assert self.extract_metric_from_df('Equal Opportunity', df, attr) == 0
        assert self.extract_metric_from_df('FNR difference', df, attr) == 0
        assert self.extract_metric_from_df('Generalized Entropy Index', df, attr) == 0
        assert self.extract_metric_from_df('Predictive Equality', df, attr) == 0
        assert self.extract_metric_from_df('Statistical Parity', df, attr) == 0
        assert self.extract_metric_from_df('Theil Index', df, attr) == 0

        attr = 'Lower Bound'
        assert self.extract_metric_from_df('Average Odds', df, attr) == -.2
        assert self.extract_metric_from_df('Disparate Impact', df, attr) == .8
        assert self.extract_metric_from_df('Equal Opportunity', df, attr) == -.2
        assert self.extract_metric_from_df('FNR difference', df, attr) == -.2
        assert self.extract_metric_from_df('Generalized Entropy Index', df, attr) == 0.
        assert self.extract_metric_from_df('Predictive Equality', df, attr) == -.2
        assert self.extract_metric_from_df('Statistical Parity', df, attr) == -.2
        assert self.extract_metric_from_df('Theil Index', df, attr) == 0.

        attr = 'Upper Bound'
        assert self.extract_metric_from_df('Average Odds', df, attr) == .2
        assert self.extract_metric_from_df('Disparate Impact', df, attr) == 1.2
        assert self.extract_metric_from_df('Equal Opportunity', df, attr) == .2
        assert self.extract_metric_from_df('FNR difference', df, attr) == .2
        assert not np.isfinite(self.extract_metric_from_df('Generalized Entropy Index', df, attr))
        assert self.extract_metric_from_df('Predictive Equality', df, attr) == .2
        assert self.extract_metric_from_df('Statistical Parity', df, attr) == .2
        assert not np.isfinite(self.extract_metric_from_df('Theil Index', df, attr))

    def test_all_scores_invalid(self):

        # test conversion of None
        y_true = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        is_member = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        with self.assertWarns(UserWarning):
            df = BinaryFairnessMetrics.get_all_scores(y_true, y_pred, is_member)
        assert type(df) == pd.DataFrame

        assert self.extract_metric_from_df('Statistical Parity', df) == 0.
        assert not np.isfinite(self.extract_metric_from_df('Average Odds', df))
        assert not np.isfinite(self.extract_metric_from_df('Disparate Impact', df))
        assert self.extract_metric_from_df('FNR difference', df) == 0.
        assert not np.isfinite(self.extract_metric_from_df('Predictive Equality', df))
        assert self.extract_metric_from_df('Generalized Entropy Index', df) == 2.
        assert np.isclose(self.extract_metric_from_df('Theil Index', df), 1.609, atol=0.01)

        attr = 'Ideal Value'
        assert self.extract_metric_from_df('Average Odds', df, attr) == 0
        assert self.extract_metric_from_df('Disparate Impact', df, attr) == 1
        assert self.extract_metric_from_df('Equal Opportunity', df, attr) == 0
        assert self.extract_metric_from_df('FNR difference', df, attr) == 0
        assert self.extract_metric_from_df('Generalized Entropy Index', df, attr) == 0
        assert self.extract_metric_from_df('Predictive Equality', df, attr) == 0
        assert self.extract_metric_from_df('Statistical Parity', df, attr) == 0
        assert self.extract_metric_from_df('Theil Index', df, attr) == 0

        attr = 'Lower Bound'
        assert self.extract_metric_from_df('Average Odds', df, attr) == -.2
        assert self.extract_metric_from_df('Disparate Impact', df, attr) == .8
        assert self.extract_metric_from_df('Equal Opportunity', df, attr) == -.2
        assert self.extract_metric_from_df('FNR difference', df, attr) == -.2
        assert self.extract_metric_from_df('Generalized Entropy Index', df, attr) == 0.
        assert self.extract_metric_from_df('Predictive Equality', df, attr) == -.2
        assert self.extract_metric_from_df('Statistical Parity', df, attr) == -.2
        assert self.extract_metric_from_df('Theil Index', df, attr) == 0.

        attr = 'Upper Bound'
        assert self.extract_metric_from_df('Average Odds', df, attr) == .2
        assert self.extract_metric_from_df('Disparate Impact', df, attr) == 1.2
        assert self.extract_metric_from_df('Equal Opportunity', df, attr) == .2
        assert self.extract_metric_from_df('FNR difference', df, attr) == .2
        assert not np.isfinite(self.extract_metric_from_df('Generalized Entropy Index', df, attr))
        assert self.extract_metric_from_df('Predictive Equality', df, attr) == .2
        assert self.extract_metric_from_df('Statistical Parity', df, attr) == .2
        assert not np.isfinite(self.extract_metric_from_df('Theil Index', df, attr))

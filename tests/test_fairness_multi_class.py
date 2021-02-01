# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from jurity.utils import convert_one_vs_rest
from jurity.fairness import BinaryFairnessMetrics, MultiClassFairnessMetrics


class TestMultiClassFairness(unittest.TestCase):

    def test_one_hot_encoding_multi_class(self):

        classes = ['a', 'b', 'c']
        predictions = ['a', 'b', 'c']
        multi_metric = MultiClassFairnessMetrics.DisparateImpact(list_of_classes=classes)

        one_hot_df = multi_metric._one_hot_encode_classes(predictions)
        assert one_hot_df['a'].tolist() == [1, 0, 0]
        assert one_hot_df['b'].tolist() == [0, 1, 0]
        assert one_hot_df['c'].tolist() == [0, 0, 1]

    def test_one_hot_encoding_multi_class_multi_label(self):

        classes = ['a', 'b', 'c']
        predictions = [['a', 'c'], ['b', 'c'], ['c']]
        multi_metric = MultiClassFairnessMetrics.DisparateImpact(list_of_classes=classes)

        one_hot_df = multi_metric._one_hot_encode_classes(predictions)
        assert one_hot_df['a'].tolist() == [1, 0, 0]
        assert one_hot_df['b'].tolist() == [0, 1, 0]
        assert one_hot_df['c'].tolist() == [1, 1, 1]

    def test_one_hot_with_ints(self):
        classes = [0, 1, 2]
        predictions = [0, 1, 2]

        multi_metric = MultiClassFairnessMetrics.DisparateImpact(list_of_classes=classes)

        one_hot_df = multi_metric._one_hot_encode_classes(predictions)

        assert one_hot_df[0].tolist() == [1, 0, 0]
        assert one_hot_df[1].tolist() == [0, 1, 0]
        assert one_hot_df[2].tolist() == [0, 0, 1]

    def test_one_hot_with_floats(self):
        classes = [0., 1., 2.]
        predictions = [0., 1., 2.]

        multi_metric = MultiClassFairnessMetrics.DisparateImpact(list_of_classes=classes)

        one_hot_df = multi_metric._one_hot_encode_classes(predictions)

        assert one_hot_df[0.].tolist() == [1, 0, 0]
        assert one_hot_df[1.].tolist() == [0, 1, 0]
        assert one_hot_df[2.].tolist() == [0, 0, 1]

    def test_disp_impact_normal_list(self):

        # Group membership
        is_member = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        # Predictions - 3 classes
        y_pred = ['a', 'b', 'b', 'a', 'c', 'c', 'a', 'b', 'a', 'c']

        # classes for multi-class classification
        classes = ['a', 'b', 'c']

        # Multiclass Fairness Metric
        multi_metric = MultiClassFairnessMetrics.DisparateImpact(list_of_classes=classes)

        result = multi_metric.get_scores(y_pred, is_member)

        # get one-hot encoded 0-1 like arrays for each class
        y_pred_a = convert_one_vs_rest('a', y_pred)
        y_pred_b = convert_one_vs_rest('b', y_pred)
        y_pred_c = convert_one_vs_rest('c', y_pred)

        # create a binary metric to test whether binary and multiclass give the same output
        binary_metric = BinaryFairnessMetrics.DisparateImpact()

        assert binary_metric.get_score(y_pred_a, is_member) == result[0]
        assert binary_metric.get_score(y_pred_b, is_member) == result[1]
        assert binary_metric.get_score(y_pred_c, is_member) == result[2]

    def test_disp_impact_multilabel(self):

        # Group membership
        is_member = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        # Predictions - 3 classes
        y_pred = [['a', 'b'], ['b', 'c'], ['b'], ['a', 'b'], ['c', 'a'], ['c'], ['a', 'b'], [], ['a', 'b'], ['c']]

        # classes for multi-class classification
        classes = ['a', 'b', 'c']

        # Multiclass Fairness Metric
        multi_metric = MultiClassFairnessMetrics.DisparateImpact(list_of_classes=classes)

        result = multi_metric.get_scores(y_pred, is_member)

        one_hot = multi_metric._one_hot_encode_classes(y_pred)

        binary_metric = BinaryFairnessMetrics.DisparateImpact()

        assert np.isclose(binary_metric.get_score(one_hot['a'], is_member), result[0], atol=0.001)
        assert np.isclose(binary_metric.get_score(one_hot['b'], is_member), result[1], atol=0.001)
        assert np.isclose(binary_metric.get_score(one_hot['c'], is_member), result[2], atol=0.001)

    def test_stat_parity_normal_list(self):

        # Group membership
        is_member = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        # Predictions - 3 classes
        y_pred = ['a', 'b', 'b', 'a', 'c', 'c', 'a', 'b', 'a', 'c']

        # classes for multi-class classification
        classes = ['a', 'b', 'c']

        # Multiclass Fairness Metric
        multi_metric = MultiClassFairnessMetrics.StatisticalParity(list_of_classes=classes)

        result = multi_metric.get_scores(y_pred, is_member)

        # get one-hot encoded 0-1 like arrays for each class
        y_pred_a = convert_one_vs_rest('a', y_pred)
        y_pred_b = convert_one_vs_rest('b', y_pred)
        y_pred_c = convert_one_vs_rest('c', y_pred)

        # create a binary metric to test whether binary and multiclass give the same output
        binary_metric = BinaryFairnessMetrics.StatisticalParity()

        assert binary_metric.get_score(y_pred_a, is_member) == result[0]
        assert binary_metric.get_score(y_pred_b, is_member) == result[1]
        assert binary_metric.get_score(y_pred_c, is_member) == result[2]

    def test_stat_parity_multi_label(self):

        # Group membership
        is_member = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        # Predictions - 3 classes
        y_pred = [['a', 'b'], ['b', 'c'], ['b'], ['a', 'b'], ['c', 'a'], ['c'], ['a', 'b'], [], ['a', 'b'], ['c']]

        # classes for multi-class classification
        classes = ['a', 'b', 'c']

        # Multiclass Fairness Metric
        multi_metric = MultiClassFairnessMetrics.StatisticalParity(list_of_classes=classes)

        result = multi_metric.get_scores(y_pred, is_member)

        one_hot = multi_metric._one_hot_encode_classes(y_pred)

        binary_metric = BinaryFairnessMetrics.StatisticalParity()

        assert np.isclose(binary_metric.get_score(one_hot['a'], is_member), result[0], atol=0.001)
        assert np.isclose(binary_metric.get_score(one_hot['b'], is_member), result[1], atol=0.001)
        assert np.isclose(binary_metric.get_score(one_hot['c'], is_member), result[2], atol=0.001)

    def test_inconsistent_input(self):
        classes = ['a', 1]
        y_pred = ['a', 1, 'a', 1]
        is_member = [1, 1, 0, 0]
        multi_metric = MultiClassFairnessMetrics.StatisticalParity(list_of_classes=classes)

        with self.assertRaises(TypeError):
            multi_metric.get_scores(y_pred, is_member)

    def test_inconsistent_multi_label(self):
        classes = ['a', 'b']
        y_pred = ['a', ['b'], 'a', 'b']
        is_member = [1, 1, 0, 0]
        multi_metric = MultiClassFairnessMetrics.StatisticalParity(list_of_classes=classes)

        with self.assertRaises(TypeError):
            multi_metric.get_scores(y_pred, is_member)

    def test_classes_dont_match_predictions(self):
        classes = ['a', 'b']
        y_pred = ['a', 1, 'a', 'b']
        is_member = [1, 1, 0, 0]
        multi_metric = MultiClassFairnessMetrics.StatisticalParity(list_of_classes=classes)

        with self.assertRaises(TypeError):
            multi_metric.get_scores(y_pred, is_member)

    def test_binary_matches_multiclass_disp_impact(self):
        binary_predictions = [0, 1, 0, 0, 1, 1]
        is_member = [0, 1, 1, 0, 0, 1]

        metric = BinaryFairnessMetrics.DisparateImpact()
        score = metric.get_score(binary_predictions, is_member)

        classes = [0, 1]
        multi_metric = MultiClassFairnessMetrics.DisparateImpact(list_of_classes=classes)
        multi_score = multi_metric.get_scores(binary_predictions, is_member)

        assert score == multi_score[1]

    def test_binary_matches_multiclass_stat_parity(self):

        binary_predictions = [0, 1, 0, 0, 1, 1]
        is_member = [0, 1, 1, 0, 0, 1]

        metric = BinaryFairnessMetrics.StatisticalParity()
        score = metric.get_score(binary_predictions, is_member)

        classes = [0, 1]
        multi_metric = MultiClassFairnessMetrics.StatisticalParity(list_of_classes=classes)
        multi_score = multi_metric.get_scores(binary_predictions, is_member)

        assert score == multi_score[1]

    def test_all_scores_multi_class_stat_parity(self):

        # Group membership
        is_member = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        # Predictions - 3 classes
        y_pred = ['a', 'b', 'b', 'a', 'a', 'c', 'a', 'a', 'c', 'b']

        # classes for multi-class classification
        classes = ['a', 'b', 'c']

        # Multiclass Fairness Metric
        all_scores = MultiClassFairnessMetrics.get_all_scores(predictions=y_pred,
                                                              is_member=is_member,
                                                              list_of_classes=classes)

        all_scores = all_scores.reset_index()

        # Get one-hot encoded 0-1 like arrays for each class
        y_pred_a = convert_one_vs_rest('a', y_pred)
        y_pred_b = convert_one_vs_rest('b', y_pred)
        y_pred_c = convert_one_vs_rest('c', y_pred)

        # Create a binary metric to test whether binary and multiclass give the same output
        # Statistical parity
        binary_metric = BinaryFairnessMetrics.StatisticalParity()

        binary_score_a = binary_metric.get_score(y_pred_a, is_member)
        binary_score_b = binary_metric.get_score(y_pred_b, is_member)
        binary_score_c = binary_metric.get_score(y_pred_c, is_member)

        multi_score_a = all_scores.loc[all_scores['Metric'] == 'Statistical Parity']['a'].values[0]
        multi_score_b = all_scores.loc[all_scores['Metric'] == 'Statistical Parity']['b'].values[0]
        multi_score_c = all_scores.loc[all_scores['Metric'] == 'Statistical Parity']['c'].values[0]

        self.assertAlmostEqual(binary_score_a, multi_score_a)
        self.assertAlmostEqual(binary_score_b, multi_score_b)
        self.assertAlmostEqual(binary_score_c, multi_score_c)

    def test_all_scores_multi_class_disp_impact(self):

        # Group membership
        is_member = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        # Predictions - 3 classes
        y_pred = ['a', 'b', 'b', 'a', 'a', 'c', 'a', 'a', 'c', 'b']

        # Classes for multi-class classification
        classes = ['a', 'b', 'c']

        # Multiclass Fairness Metric
        all_scores = MultiClassFairnessMetrics.get_all_scores(predictions=y_pred,
                                                              is_member=is_member,
                                                              list_of_classes=classes)
        all_scores = all_scores.reset_index()

        # Get one-hot encoded 0-1 like arrays for each class
        y_pred_a = convert_one_vs_rest('a', y_pred)
        y_pred_b = convert_one_vs_rest('b', y_pred)
        y_pred_c = convert_one_vs_rest('c', y_pred)

        # Create a binary metric to test whether binary and multiclass give the same output
        # Disparate impact
        binary_metric = BinaryFairnessMetrics.DisparateImpact()

        binary_score_a = binary_metric.get_score(y_pred_a, is_member)
        binary_score_b = binary_metric.get_score(y_pred_b, is_member)
        binary_score_c = binary_metric.get_score(y_pred_c, is_member)

        multi_score_a = all_scores.loc[all_scores['Metric'] == 'Disparate Impact']['a'].values[0]
        multi_score_b = all_scores.loc[all_scores['Metric'] == 'Disparate Impact']['b'].values[0]
        multi_score_c = all_scores.loc[all_scores['Metric'] == 'Disparate Impact']['c'].values[0]

        self.assertAlmostEqual(binary_score_a, multi_score_a)
        self.assertAlmostEqual(binary_score_b, multi_score_b)
        self.assertAlmostEqual(binary_score_c, multi_score_c)

    def test_all_scores_multi_class_multi_label_stat_parity(self):

        # Group membership
        is_member = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        # Predictions - 3 classes
        y_pred = [['a', 'b'], ['b', 'c'], ['b'], ['a', 'b'], ['c', 'a'], ['c'], ['a', 'b'], [], ['a', 'b'], ['c']]

        # Classes for multi-class classification
        classes = ['a', 'b', 'c']

        # Multiclass Fairness Metric
        all_scores = MultiClassFairnessMetrics.get_all_scores(predictions=y_pred,
                                                              is_member=is_member,
                                                              list_of_classes=classes)
        all_scores = all_scores.reset_index()

        # Get one-hot encoded 0-1 like arrays for each class
        mlb = MultiLabelBinarizer(classes=classes)
        predictions = pd.Series(y_pred)

        one_hot = pd.DataFrame(
                mlb.fit_transform(predictions),
                columns=mlb.classes_,
                index=predictions.index,
            )

        y_pred_a = one_hot['a']
        y_pred_b = one_hot['b']
        y_pred_c = one_hot['c']

        # Create a binary metric to test whether binary and multiclass give the same output
        # Statistical parity
        binary_metric = BinaryFairnessMetrics.StatisticalParity()

        binary_score_a = binary_metric.get_score(y_pred_a, is_member)
        binary_score_b = binary_metric.get_score(y_pred_b, is_member)
        binary_score_c = binary_metric.get_score(y_pred_c, is_member)

        multi_score_a = all_scores.loc[all_scores['Metric'] == 'Statistical Parity']['a'].values[0]
        multi_score_b = all_scores.loc[all_scores['Metric'] == 'Statistical Parity']['b'].values[0]
        multi_score_c = all_scores.loc[all_scores['Metric'] == 'Statistical Parity']['c'].values[0]

        self.assertAlmostEqual(binary_score_a, multi_score_a)
        self.assertAlmostEqual(binary_score_b, multi_score_b)
        self.assertAlmostEqual(binary_score_c, multi_score_c)

    def test_all_scores_multi_class_multi_label_disp_impact(self):

        # Group membership
        is_member = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        # Predictions - 3 classes
        y_pred = [['a', 'b'], ['b', 'c'], ['b'], ['a', 'b'], ['c', 'a'], ['c'], ['a', 'b'], [], ['a', 'b'], ['c']]

        # classes for multi-class classification
        classes = ['a', 'b', 'c']

        # Multiclass Fairness Metric
        all_scores = MultiClassFairnessMetrics.get_all_scores(predictions=y_pred,
                                                              is_member=is_member,
                                                              list_of_classes=classes)
        all_scores = all_scores.reset_index()

        # Get one-hot encoded 0-1 like arrays for each class
        mlb = MultiLabelBinarizer(classes=classes)
        predictions = pd.Series(y_pred)

        one_hot = pd.DataFrame(
                mlb.fit_transform(predictions),
                columns=mlb.classes_,
                index=predictions.index,
            )

        y_pred_a = one_hot['a']
        y_pred_b = one_hot['b']
        y_pred_c = one_hot['c']

        # Create a binary metric to test whether binary and multiclass give the same output
        # Disparate impact
        binary_metric = BinaryFairnessMetrics.DisparateImpact()

        binary_score_a = binary_metric.get_score(y_pred_a, is_member)
        binary_score_b = binary_metric.get_score(y_pred_b, is_member)
        binary_score_c = binary_metric.get_score(y_pred_c, is_member)

        multi_score_a = all_scores.loc[all_scores['Metric'] == 'Disparate Impact']['a'].values[0]
        multi_score_b = all_scores.loc[all_scores['Metric'] == 'Disparate Impact']['b'].values[0]
        multi_score_c = all_scores.loc[all_scores['Metric'] == 'Disparate Impact']['c'].values[0]

        self.assertAlmostEqual(binary_score_a, multi_score_a)
        self.assertAlmostEqual(binary_score_b, multi_score_b)
        self.assertAlmostEqual(binary_score_c, multi_score_c)

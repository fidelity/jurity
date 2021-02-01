# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pandas as pd

from jurity.classification import BinaryClassificationMetrics


class TestClassificationMetrics(unittest.TestCase):

    def test_accuracy(self):

        # Data
        actual = [1, 1, 0, 1, 0, 0]
        predicted = [1, 1, 0, 1, 0, 0]

        # Metric
        metric = BinaryClassificationMetrics.Accuracy()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 1)

    def test_accuracy_numpy(self):

        # Data
        actual = np.array([1, 1, 0, 1, 0, 0])
        predicted = np.array([0, 0, 0, 0, 0, 0])

        # Metric
        metric = BinaryClassificationMetrics.Accuracy()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 0.5)

    def test_accuracy_pandas(self):

        # Data
        actual = pd.Series([0, 0, 0, 0, 0, 0])
        predicted = pd.Series([0, 0, 0, 0, 0, 0])

        # Metric
        metric = BinaryClassificationMetrics.Accuracy()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 1)

    def test_accuracy_non_binary_input(self):

        # Data
        actual = [0, 1, 2, 0, 0, 0]
        predicted = [0, 0, 0, 0, 0, 0]

        # Metric
        metric = BinaryClassificationMetrics.Accuracy()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, predicted)

    def test_accuracy_non_zero_one_input(self):

        # Data
        actual = ['a', 'b', 'a', 'a']
        predicted = ['a', 'b', 'a', 'a']

        # Metric
        metric = BinaryClassificationMetrics.Accuracy()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, predicted)

    def test_auc(self):

        # Data
        actual = [1, 1, 0, 1, 0, 0]
        likelihoods = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        # Metric
        metric = BinaryClassificationMetrics.AUC()

        # Score
        score = metric.get_score(actual, likelihoods)
        self.assertEqual(score, 0.5)

    def test_auc_numpy(self):

        # Data
        actual = np.array([1, 1, 0, 1, 0, 0])
        likelihoods = np.array([1, 1, 1, 0.5, 0.5, 0.5])

        # Metric
        metric = BinaryClassificationMetrics.AUC()

        # Score
        score = metric.get_score(actual, likelihoods)
        self.assertEqual(score, 0.6666666666666667)

    def test_auc_pandas(self):

        # Data
        actual = pd.Series([0, 0, 0, 0, 0, 1])
        likelihoods = pd.Series([0, 0, 0, 0.5, 0.5, 0.5])

        # Metric
        metric = BinaryClassificationMetrics.AUC()

        # Score
        score = metric.get_score(actual, likelihoods)
        self.assertEqual(score, 0.8)

    def test_auc_non_binary_input(self):

        # Data
        actual = [0, 1, 2, 0, 0, 0]
        likelihoods = [0, 0, 0, 0.5, 0.5, 0.5]

        # Metric
        metric = BinaryClassificationMetrics.AUC()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, likelihoods)

    def test_auc_non_zero_one_input(self):

        # Data
        actual = ['a', 'b', 'a', 'a']
        likelihoods = [0, 0, 0, 0.5, 0.5, 0.5]

        # Metric
        metric = BinaryClassificationMetrics.AUC()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, likelihoods)

    def test_auc_likelihood_input(self):

        # Data
        actual = [1, 1, 0, 1, 0, 0]
        likelihoods = [100, 0, 0, 0.5, 0.5, 0.5]

        # Metric
        metric = BinaryClassificationMetrics.AUC()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, likelihoods)

    def test_f1(self):

        # Data
        actual = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1]
        predicted = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]

        # Metric
        metric = BinaryClassificationMetrics.F1()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 0.6666666666666666)

    def test_f1_numpy(self):

        # Data
        actual = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1])
        predicted = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0])

        # Metric
        metric = BinaryClassificationMetrics.F1()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 0.6666666666666666)

    def test_f1_pandas(self):

        # Data
        actual = pd.Series([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1])
        predicted = pd.Series([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0])

        # Metric
        metric = BinaryClassificationMetrics.F1()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 0.6666666666666666)

    def test_f1_non_binary_input(self):

        # Data
        actual = [0, 1, 2, 0, 0, 0]
        predicted = [0, 0, 0, 0, 0, 0]

        # Metric
        metric = BinaryClassificationMetrics.F1()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, predicted)

    def test_f1_non_zero_one_input(self):

        # Data
        actual = ['a', 'b', 'a', 'a']
        predicted = ['a', 'b', 'a', 'a']

        # Metric
        metric = BinaryClassificationMetrics.F1()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, predicted)

    def test_classification_precision(self):
        # Data
        actual = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1]
        predicted = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]

        # Metric
        metric = BinaryClassificationMetrics.Precision()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 0.7142857142857143)

    def test_classification_precision_numpy(self):
        # Data
        actual = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1])
        predicted = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0])

        # Metric
        metric = BinaryClassificationMetrics.Precision()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 0.7142857142857143)

    def test_classification_precision_pandas(self):
        # Data
        actual = pd.Series([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1])
        predicted = pd.Series([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0])

        # Metric
        metric = BinaryClassificationMetrics.Precision()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 0.7142857142857143)

    def test_classification_precision_non_binary_input(self):
        # Data
        actual = [0, 1, 2, 0, 0, 0]
        predicted = [0, 0, 0, 0, 0, 0]

        # Metric
        metric = BinaryClassificationMetrics.Precision()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, predicted)

    def test_classification_precision_non_zero_one_input(self):
        # Data
        actual = ['a', 'b', 'a', 'a']
        predicted = ['a', 'b', 'a', 'a']

        # Metric
        metric = BinaryClassificationMetrics.Precision()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, predicted)

    def test_classification_recall(self):
        # Data
        actual = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1]
        predicted = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]

        # Metric
        metric = BinaryClassificationMetrics.Recall()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 0.75)

    def test_classification_recall_numpy(self):
        # Data
        actual = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1])
        predicted = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1])

        # Metric
        metric = BinaryClassificationMetrics.Recall()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 0.75)

    def test_classification_recall_pandas(self):
        # Data
        actual = pd.Series([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1])
        predicted = pd.Series([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1])

        # Metric
        metric = BinaryClassificationMetrics.Recall()

        # Score
        score = metric.get_score(actual, predicted)
        self.assertEqual(score, 0.75)

    def test_classification_recall_non_binary_input(self):
        # Data
        actual = [0, 1, 2, 0, 0, 0]
        predicted = [0, 0, 0, 0, 0, 0]

        # Metric
        metric = BinaryClassificationMetrics.Recall()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, predicted)

    def test_classification_recall_non_zero_one_input(self):
        # Data
        actual = ['a', 'b', 'a', 'a']
        predicted = ['a', 'b', 'a', 'a']

        # Metric
        metric = BinaryClassificationMetrics.Recall()

        # Score
        with self.assertRaises(ValueError):
            metric.get_score(actual, predicted)

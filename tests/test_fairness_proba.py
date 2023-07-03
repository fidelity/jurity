# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pandas as pd

from jurity.fairness import BinaryFairnessMetrics
from jurity.utils import InputShapeError


class TestBinaryProbFairness(unittest.TestCase):

    def test_prob_stat_parity_list(self):

        # Data
        predictions = [1, 1, 0, 1]
        memberships = [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]]
        surrogates = [0, 2, 0, 3]
        membership_labels = [1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        score = metric.get_score(predictions, memberships, surrogates, membership_labels)

    def test_prob_stat_parity_np(self):

        # Data
        predictions = np.array([1, 1, 0, 1])
        memberships = np.array([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]])
        surrogates = np.array([0, 2, 0, 3])
        membership_labels = [1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        score = metric.get_score(predictions, memberships, surrogates, membership_labels)

    def test_prob_stat_parity_df(self):

        # Data
        df = pd.DataFrame.from_dict({'predictions': [1, 1, 0, 1],
                                     'memberships': [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]],
                                     'surrogates': [0, 2, 0, 3]})
        membership_labels = [1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        score = metric.get_score(df["predictions"], df["memberships"], df["surrogates"], membership_labels)

    def test_prob_stat_parity_array_size_invalid(self):

        # Data
        predictions = np.array([1, 1, 0, 1, 0])
        memberships = np.array([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]])
        surrogates = np.array([0, 2, 0, 3])
        membership_labels = [1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        with self.assertRaises(InputShapeError):
            metric.get_score(predictions, memberships, surrogates, membership_labels)

    def test_prob_stat_parity_list_size_invalid(self):

        # Data
        predictions = [1, 1, 0, 1, 0]
        memberships = [[0.2, 0.8], [0.4, 0.6],[0.7, 0.3], [0.9, 0.1]]
        surrogates = [0, 2, 0, 3]
        membership_labels = [1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        with self.assertRaises(InputShapeError):
            metric.get_score(predictions, memberships, surrogates, membership_labels)

    def test_prob_stat_parity_df_size_invalid(self):

        # Data
        df = pd.DataFrame.from_dict({'predictions': [1, 1, 0, 1],
                                     'memberships': [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]],
                                     'surrogates': [0, 2, 0, 3]})
        membership_labels = [1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        with self.assertRaises(ValueError):
            score = metric.get_score(df["predictions"], df["memberships"], df["surrogates"], membership_labels)

    def test_prob_stat_parity_likelihood_size_invalid_outer(self):

        # Data
        predictions = np.array([1, 1, 0, 1])
        surrogates = np.array([0, 2, 0, 1])
        memberships = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        membership_labels = [1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Invalid: 4 samples, only 3 memberships
        with self.assertRaises(InputShapeError):
            metric.get_score(predictions, memberships, surrogates, membership_labels)

    def test_prob_stat_parity_likelihood_size_invalid_inner(self):

        # Data
        predictions = np.array([1, 1, 0, 1])
        surrogates = np.array([0, 2, 0, 1])
        memberships = np.array([[0.5, 0.5, 0.5], [0.5],
                                [0.5, 0.5], [0.5, 0.5]])
        membership_labels = [1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Invalid: inner memberships array not identical size
        with self.assertRaises(InputShapeError):
            metric.get_score(predictions, memberships, surrogates, membership_labels)

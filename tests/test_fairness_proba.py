# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pandas as pd
import inspect

from jurity.fairness import BinaryFairnessMetrics
from jurity.utils import InputShapeError, Constants


def run_all_fairness(labels, predictions, memberships, surrogates, membership_labels):
    fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]
    for f in fairness_funcs:
        name = f[0]
        if name not in Constants.bootstrap_implemented:
            continue
        class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
        instance = class_()  # dynamically instantiate such class
        if name in Constants.no_labels:
            score = instance.get_score(predictions, memberships, surrogates, membership_labels, None)
        else:
            score = instance.get_score(labels, predictions, memberships, surrogates, membership_labels, None)
        return score


class TestBinaryProbFairness(unittest.TestCase):

    def test_prob_list(self):
        # Data
        labels = [0, 1, 0, 1]
        predictions = [1, 1, 0, 1]
        memberships = [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]]
        surrogates = [0, 2, 0, 3]
        membership_labels = [1]
        run_all_fairness(labels, predictions, memberships, surrogates, membership_labels)

    def test_prob_stat_np(self):
        # Data
        labels = [0, 1, 0, 1]
        predictions = np.array([1, 1, 0, 1])
        memberships = np.array([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]])
        surrogates = np.array([0, 2, 0, 3])
        membership_labels = [1]

        # Metric
        metric = BinaryFairnessMetrics.StatisticalParity()

        # Score
        score = metric.get_score(predictions, memberships, surrogates, membership_labels)

    def test_prob_df(self):
        # Data
        df = pd.DataFrame.from_dict({'labels': [1, 0, 1, 1],
                                     'predictions': [1, 1, 0, 1],
                                     'memberships': [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]],
                                     'surrogates': [0, 2, 0, 3]})
        membership_labels = [1]
        run_all_fairness(df["labels"], df["predictions"], df["memberships"], df["surrogates"], membership_labels)

    def test_prob_array_size_invalid(self):
        # Data
        labels = [0, 1, 0, 1]
        predictions = np.array([1, 1, 0, 1, 0])
        memberships = np.array([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]])
        surrogates = np.array([0, 2, 0, 3])
        membership_labels = [1]

        fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]

        for f in fairness_funcs:
            name = f[0]
            if name not in Constants.bootstrap_implemented:
                continue
            class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
            instance = class_()  # dynamically instantiate such class
            if name in Constants.no_labels:
                with self.assertRaises(InputShapeError):
                    score = instance.get_score(predictions, memberships, surrogates, membership_labels, None)
            else:
                with self.assertRaises(InputShapeError):
                    score = instance.get_score(labels, predictions, memberships, surrogates, membership_labels, None)

    def test_prob_stat_parity_list_size_invalid(self):
        # Data
        labels = [0, 1, 0, 1]
        predictions = [1, 1, 0, 1, 0]
        memberships = [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]]
        surrogates = [0, 2, 0, 3]
        membership_labels = [1]

        # Metric
        fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]

        for f in fairness_funcs:
            name = f[0]
            if name not in Constants.bootstrap_implemented:
                continue
            class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
            instance = class_()  # dynamically instantiate such class
            if name in Constants.no_labels:
                with self.assertRaises(InputShapeError):
                    score = instance.get_score(predictions, memberships, surrogates, membership_labels, None)
            else:
                with self.assertRaises(InputShapeError):
                    score = instance.get_score(labels, predictions, memberships, surrogates, membership_labels, None)

    def test_prob_df_size_invalid(self):
        # Data
        df = pd.DataFrame.from_dict({'predictions': [1, 1, 0, 1, 1]})
        df2 = pd.DataFrame.from_dict({'labels': [0, 1, 0, 1],
                                      'memberships': [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]],
                                      'surrogates': [0, 2, 0, 3]})
        membership_labels = [1]

        fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]

        for f in fairness_funcs:
            name = f[0]
            if name not in Constants.bootstrap_implemented:
                continue
            class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
            instance = class_()  # dynamically instantiate such class
            if name in Constants.no_labels:
                with self.assertRaises(InputShapeError, \
                                       msg=f"{name} does not raise InputShapeError on invalid predictions shape."):
                    score = instance.get_score(df["predictions"], df2["memberships"], df2["surrogates"],
                                               membership_labels)
            else:
                with self.assertRaises(InputShapeError, msg=f"{name} does not raise InputShapeError on invalid predictions shape"):
                    score = instance.get_score(df2["labels"],df["predictions"], df2["memberships"], df2["surrogates"],
                                               membership_labels)

    def test_prob_likelihood_size_invalid_outer(self):
        # Data
        labels = [0, 1, 0, 1]
        predictions = np.array([1, 1, 0, 1])
        surrogates = np.array([0, 2, 0, 1])
        memberships = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        membership_labels = [1]

        fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]

        for f in fairness_funcs:
            name = f[0]
            if name not in Constants.bootstrap_implemented:
                continue
            class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
            instance = class_()  # dynamically instantiate such class
            if name not in Constants.bootstrap_implemented:
                continue
            if name in Constants.no_labels:
                with self.assertRaises(InputShapeError,
                                       msg=f"{name} does not raise shape error on invalid likelihood outer shape"):
                    instance.get_score(predictions, memberships, surrogates, membership_labels)
            else:
                with self.assertRaises(InputShapeError,
                                       msg=f"{name} does not raise shape error on invalid likelihood outer shape"):
                    instance.get_score(labels,predictions, memberships, surrogates, membership_labels)


    def test_prob_likelihood_size_invalid_inner(self):
        # Data
        labels = [0, 1, 0, 1]
        predictions = np.array([1, 1, 0, 1])
        surrogates = np.array([0, 2, 0, 1])
        memberships = np.array([[0.5, 0.5, 0.5], [0.5],
                                [0.5, 0.5], [0.5, 0.5]])
        membership_labels = [1]

        fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]

        for f in fairness_funcs:
            name = f[0]
            if name not in Constants.bootstrap_implemented:
                continue
            class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
            instance = class_()  # dynamically instantiate such class
            if name in Constants.no_labels:
                with self.assertRaises(InputShapeError,
                                       msg=f"{name} does not raise shape error on invalid inner likelihood shape"):
                    instance.get_score(predictions, memberships, surrogates, membership_labels)
            else:
                with self.assertRaises(InputShapeError,
                                       msg=f"{name} does not raise shape error on invalid inner likelihood shape"):
                    instance.get_score(labels, predictions, memberships, surrogates, membership_labels)
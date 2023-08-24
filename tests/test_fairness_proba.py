# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pandas as pd
import inspect

from jurity.fairness import BinaryFairnessMetrics
from jurity.utils import InputShapeError, Constants

class TestBinaryProbFairness(unittest.TestCase):
    def run_one_score(self,instance, labels, predictions, memberships, surrogates, membership_labels):
        if labels is None:
            score = instance.get_score(predictions, memberships, surrogates, membership_labels, None)
        else:
            score = instance.get_score(labels, predictions, memberships, surrogates, membership_labels, None)
        return score

    def run_all_fairness(self,labels, predictions, memberships, surrogates, membership_labels, error_type=None,
                         error_msg=""):
        """
        Run all probabilistic fairness metrics that are implemented according to Constants.bootstrap_implemented
        Check for errors when error_type is not None
        """
        fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]
        for f in fairness_funcs:
            name = f[0]
            if name not in Constants.bootstrap_implemented:
                continue
            class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
            instance = class_()  # dynamically instantiate such class

            if name in Constants.no_labels: #handle cases where there are no labels
                these_labels = None
            else:
                these_labels = labels

            if error_type is not None:
                with self.assertRaises(error_type, msg=f"{name} {error_msg}"):
                    score = self.run_one_score(instance, these_labels, predictions, memberships, surrogates, membership_labels)
            else:
                score = self.run_one_score(instance, these_labels, predictions, memberships, surrogates, membership_labels)

    def test_prob_list(self):
        "All return a score when the input membership is a list"
        # Data
        labels = [0, 1, 0, 1]
        predictions = [1, 1, 0, 1]
        memberships = [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]]
        surrogates = [0, 2, 0, 3]
        membership_labels = [1]
        self.run_all_fairness(labels, predictions, memberships, surrogates, membership_labels)

    def test_prob_df(self):
        "All return a score when the inputs are from the same dataframe"
        # Data
        df = pd.DataFrame.from_dict({'labels': [1, 0, 1, 1],
                                     'predictions': [1, 1, 0, 1],
                                     'memberships': [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]],
                                     'surrogates': [0, 2, 0, 3]})
        membership_labels = [1]
        self.run_all_fairness(df["labels"], df["predictions"], df["memberships"], df["surrogates"], membership_labels)

    def test_prob_surrogate_size_invalid(self):

        # Data
        labels = [0, 1, 0, 1]
        predictions = np.array([1, 1, 0, 1, 0])
        memberships = np.array([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]])
        surrogates = np.array([0, 2, 0, 3, 1])
        membership_labels = [1]
        self.run_all_fairness(labels, predictions, memberships, surrogates, membership_labels,InputShapeError)

    def test_prob_df_size_invalid(self):
        # Data
        df = pd.DataFrame.from_dict({'predictions': [1, 1, 0, 1, 1]})
        df2 = pd.DataFrame.from_dict({'labels': [0, 1, 0, 1],
                                      'memberships': [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]],
                                      'surrogates': [0, 2, 0, 3]})
        membership_labels = [1]
        self.run_all_fairness(df2["labels"],df["predictions"],df2["memberships"],df2["surrogates"],membership_labels,
                              InputShapeError,"does not raise InputShapeError on invalid predictions shape")

    def test_prob_likelihood_size_invalid_outer(self):
        # Data
        labels = [0, 1, 0, 1]
        predictions = np.array([1, 1, 0, 1])
        surrogates = np.array([0, 2, 0, 1])
        memberships = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        membership_labels = [1]
        self.run_all_fairness(labels, predictions, memberships, surrogates, membership_labels,
                              InputShapeError,"does not raise shape error on invalid likelihood outer shape")

    def test_prob_likelihood_size_invalid_inner(self):
        # Data
        labels = [0, 1, 0, 1]
        predictions = np.array([1, 1, 0, 1])
        surrogates = np.array([0, 2, 0, 1])
        memberships = np.array([list([0.5, 0.5, 0.5]), list([0.5]),
                                list([0.5, 0.5]), list([0.5, 0.5])],dtype=object)
        membership_labels = [1]
        self.run_all_fairness(labels, predictions, memberships, surrogates, membership_labels,
                              InputShapeError,"does not raise shape error on invalid inner likelihood shape")
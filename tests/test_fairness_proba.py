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
        """
        Helper function to run the appropriate score metric, depending on inputs
        """
        if labels is None:
            score = instance.get_score(predictions, memberships, surrogates, membership_labels, None)
        else:
            score = instance.get_score(labels, predictions, memberships, surrogates, membership_labels, None)
        return score

    def run_all_fairness(self,labels, predictions, memberships, surrogates, membership_labels, error_type=None,
                         error_msg=""):
        """
        Helper function to run all probabilistic fairness metrics that are implemented according to Constants.bootstrap_implemented
        Checks for errors when error_type is not None.
        Otherwise ensures that scores are returned
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
        """
        Ensure all return a score when the inputs are from the same dataframe
        """
        # Data
        df = pd.DataFrame.from_dict({'labels': [1, 0, 1, 1],
                                     'predictions': [1, 1, 0, 1],
                                     'memberships': [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]],
                                     'surrogates': [0, 2, 0, 3]})
        membership_labels = [1]
        self.run_all_fairness(df["labels"], df["predictions"], df["memberships"], df["surrogates"], membership_labels)

    def test_prob_surrogate_size_invalid(self):
        """
        Ensure the errors are returned for invalid shape.
        """
        # Data
        labels = [0, 1, 0, 1]
        predictions = np.array([1, 1, 0, 1, 0])
        memberships = np.array([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]])
        surrogates = np.array([0, 2, 0, 3, 1])
        membership_labels = [1]
        self.run_all_fairness(labels, predictions, memberships, surrogates, membership_labels,InputShapeError)

    def test_prob_df_size_invalid(self):
        """
        Ensure invalid inout dataframe for memberships returns an error
        """
        # Data
        df = pd.DataFrame.from_dict({'predictions': [1, 1, 0, 1, 1]})
        df2 = pd.DataFrame.from_dict({'labels': [0, 1, 0, 1],
                                      'memberships': [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]],
                                      'surrogates': [0, 2, 0, 3]})
        membership_labels = [1]
        self.run_all_fairness(df2["labels"],df["predictions"],df2["memberships"],df2["surrogates"],membership_labels,
                              InputShapeError,"does not raise InputShapeError on invalid predictions shape")

    def test_prob_likelihood_size_invalid_outer(self):
        """
        Ensure invalid outer shape for memberships returns an error
        """
        # Data
        labels = [0, 1, 0, 1]
        predictions = np.array([1, 1, 0, 1])
        surrogates = np.array([0, 2, 0, 1])
        memberships = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        membership_labels = [1]
        self.run_all_fairness(labels, predictions, memberships, surrogates, membership_labels,
                              InputShapeError,"does not raise shape error on invalid likelihood outer shape")

    def test_prob_likelihood_size_invalid_inner(self):
        """
        Ensure invalid inner shape for memberhsips returns an error
        """
        # Data
        labels = [0, 1, 0, 1]
        predictions = np.array([1, 1, 0, 1])
        surrogates = np.array([0, 2, 0, 1])
        memberships = np.array([list([0.5, 0.5, 0.5]), list([0.5]),
                                list([0.5, 0.5]), list([0.5, 0.5])],dtype=object)
        membership_labels = [1]
        self.run_all_fairness(labels, predictions, memberships, surrogates, membership_labels,
                              InputShapeError,"does not raise shape error on invalid inner likelihood shape")

    def test_boot_results_input(self):
        """
        Make all score functions unpack results correctly from pre-generated bootstrap.
        """
        #Construct sample bootstrap dpandas.DataFrame
        c = pd.Series(["W", "NW"], name="class")
        answer_dict = {"FPR": [0.6, 0.689655],
                       "FNR": [0.5, 0.985915],
                       "TPR": [0.5, 0.014085],
                       "TNR": [0.4, 0.310345],
                       "ACC": [0.45, 0.10],
                       "Prediction Rate":[0.55,0.21]}
        raw_boot_results = pd.DataFrame({
            "false_negative_ratio": [0.25, 0.7],
            "false_positive_ratio": [0.3, 0.2],
            "true_negative_ratio": [0.2, 0.09],
            "true_positive_ratio": [0.25, 0.01],
            "prediction_ratio": [0.55, 0.21]})
        test_boot_results = pd.concat([pd.DataFrame.from_dict(answer_dict), raw_boot_results], axis=1).set_index(c)
        tests=[("StatisticalParity","Prediction Rate"),
               ("PredictiveEquality","FPR"),
               ("EqualOpportunity","TPR"),
               ("FNRDifference","FNR")]
        for t in tests:
            name=t[0]
            #There are all tests that are a simple subtraction
            answer=answer_dict[t[1]][1]-answer_dict[t[1]][0]
            class_ = getattr(BinaryFairnessMetrics, t[0])  # grab a class which is a property of BinaryFairnessMetrics
            instance = class_()  # dynamically instantiate such class
            if name in Constants.no_labels:
                score = instance.get_score(None, None, None, [1], test_boot_results)
            else:
                score = instance.get_score(None, None, None, None, [1], test_boot_results)

            self.assertEqual(score, answer,
                                   f"Score for returned for {name} from bootstrap dataframe is incorrect.\nexpected: {answer}, got: {score}.")
        name="AverageOdds"
        answer=0.5*((answer_dict["FPR"][1]-answer_dict["FPR"][0])+(answer_dict["TPR"][1]-answer_dict["TPR"][0]))
        class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
        instance = class_()  # dynamically instantiate such class
        score=instance.get_score(None,None,None,None,[1],test_boot_results)
        self.assertEqual(score,answer,f"score function {name} give unexpected answer for pre-generated bootstrap. Expected: {answer}. Got {score}.")




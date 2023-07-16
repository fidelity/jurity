# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import List, Union
from typing import NamedTuple

import numpy as np
import pandas as pd

from jurity.fairness.base import _BaseBinaryFairness
from jurity.fairness.base import _BaseMultiClassMetric
from jurity.utils import check_inputs_validity
from .average_odds import AverageOdds
from .disparate_impact import BinaryDisparateImpact, MultiDisparateImpact
from .equal_opportunity import EqualOpportunity
from .fnr_difference import FNRDifference
from .for_difference import FORDifference
from .generalized_entropy import GeneralizedEntropyIndex
from .predictive_equality import PredictiveEquality
from .statistical_parity import BinaryStatisticalParity, MultiStatisticalParity
from .theil_index import TheilIndex


class BinaryFairnessMetrics(NamedTuple):
    """
    Class containing a variety of fairness metrics for binary classification.
    """

    AverageOdds = AverageOdds
    DisparateImpact = BinaryDisparateImpact
    EqualOpportunity = EqualOpportunity
    FNRDifference = FNRDifference
    FORDifference = FORDifference
    GeneralizedEntropyIndex = GeneralizedEntropyIndex
    PredictiveEquality = PredictiveEquality
    StatisticalParity = BinaryStatisticalParity
    TheilIndex = TheilIndex

    @staticmethod
    def get_all_scores(labels: Union[List, np.ndarray, pd.Series],
                       predictions: Union[List, np.ndarray, pd.Series],
                       is_member: Union[List, np.ndarray, pd.Series],
                       membership_label: Union[str, float, int] = 1) -> pd.DataFrame:
        """
        Calculates and tabulates all of the fairness metric scores.

        Parameters
        ----------
        labels: Union[List, np.ndarray, pd.Series]
            Binary ground truth labels for the provided dataset (0/1).
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        is_member: Union[List, np.ndarray, pd.Series]
            Binary membership labels (0/1).
        membership_label: Union[str, float, int]
            Value indicating group membership.
            Default value is 1.

        Returns
        ----------
        Pandas data frame with all implemented binary fairness metrics.
        """
        # Logic to check input types
        check_inputs_validity(labels=labels, predictions=predictions, is_member=is_member, optional_labels=False)

        fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]

        df = pd.DataFrame(columns=["Metric", "Value", "Ideal Value", "Lower Bound", "Upper Bound"])
        for fairness_func in fairness_funcs:

            name = fairness_func[0]
            class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
            instance = class_()  # dynamically instantiate such class

            if name in ["DisparateImpact", "StatisticalParity"]:
                score = instance.get_score(predictions, is_member, membership_label)
            elif name in ["GeneralizedEntropyIndex", "TheilIndex"]:
                score = instance.get_score(labels, predictions)
            else:
                score = instance.get_score(labels, predictions, is_member, membership_label)

            if score is None:
                score = np.nan
            score = np.round(score, 3)
            df = pd.concat([df, pd.DataFrame(
                [[instance.name, score, instance.ideal_value, instance.lower_bound, instance.upper_bound]],
                columns=df.columns)], axis=0, ignore_index=True)

        df = df.set_index("Metric")

        return df


class MultiClassFairnessMetrics(NamedTuple):
    """
    Class containing a variety of fairness metrics for multi-class classification.
    """

    DisparateImpact = MultiDisparateImpact
    StatisticalParity = MultiStatisticalParity

    @staticmethod
    def get_all_scores(predictions: Union[List, np.ndarray, pd.Series],
                       is_member: Union[List, np.ndarray, pd.Series],
                       list_of_classes: List[str]):
        """
        Calculates and tabulates all of the fairness metric scores.

        Parameters
        ----------
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        is_member: Union[List, np.ndarray, pd.Series]
            Binary membership labels (0/1).
        list_of_classes: List[str]
            List with class labels.

        Returns
        ----------
        Pandas data frame with all implemented multi-class fairness metrics.
        """

        fairness_funcs = inspect.getmembers(MultiClassFairnessMetrics, predicate=inspect.isclass)[:-1]

        df = pd.DataFrame()
        one_hot_predictions = None
        for idx, fairness_func in enumerate(fairness_funcs):

            name = fairness_func[0]

            # Grab a class which is a property of MulticlassFairnessMetrics
            multi_class = getattr(MultiClassFairnessMetrics, name)

            # Dynamically instantiate such class
            multi_instance = multi_class(list_of_classes=list_of_classes)

            if idx == 0:
                df = pd.DataFrame(columns=["Metric"] + list_of_classes + ["Ideal Value", "Lower Bound", "Upper Bound"])
                one_hot_predictions = multi_instance._one_hot_encode_classes(predictions)

            if name in ["DisparateImpact", "StatisticalParity"]:
                scores = []
                # Get score for each one-hot encoded class
                for class_ in list_of_classes:
                    score = multi_instance._binary_score(one_hot_predictions[class_], is_member)
                    scores.append(score)
                scores = [np.round(score, 3) for score in scores]
            else:
                scores = [None for _ in range(len(list_of_classes))]

            append_dict = {"Metric": multi_instance.name}
            for position, class_ in enumerate(list_of_classes):
                append_dict[class_] = scores[position]

            append_dict["Lower Bound"] = multi_instance.lower_bound
            append_dict["Ideal Value"] = multi_instance.ideal_value
            append_dict["Upper Bound"] = multi_instance.upper_bound

            df = pd.concat([df, pd.DataFrame([append_dict])], ignore_index=True)

        df = df.set_index("Metric")
        return df

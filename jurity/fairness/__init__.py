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
from jurity.utils import Constants
from jurity.utils import check_inputs, is_one_dimensional, check_inputs_proba
from jurity.utils_proba import get_bootstrap_results
from .average_odds import AverageOdds
from .disparate_impact import BinaryDisparateImpact, MultiDisparateImpact
from .equal_opportunity import EqualOpportunity
from .fnr_difference import FNRDifference
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
    GeneralizedEntropyIndex = GeneralizedEntropyIndex
    PredictiveEquality = PredictiveEquality
    StatisticalParity = BinaryStatisticalParity
    TheilIndex = TheilIndex

    @staticmethod
    def get_all_scores(labels: Union[List, np.ndarray, pd.Series],
                       predictions: Union[List, np.ndarray, pd.Series],
                       memberships: Union[List, np.ndarray, pd.Series],
                       surrogates: Union[List, np.ndarray, pd.Series] = None,
                       membership_labels: Union[str, float, int, List, np.array] = 1) -> pd.DataFrame:
        """
        Calculates and tabulates all fairness metric scores.
        Parameters
        ----------
        labels: Union[List, np.ndarray, pd.Series]
            Binary ground truth labels for each sample.
        predictions: Union[List, np.ndarray, pd.Series]
            Binary prediction for each sample from a black-box classifier binary (0/1).
        memberships: Union[List, np.ndarray, pd.Series, List[List], pd.DataFrame]
            Membership attribute for each sample.
                If deterministic, it is the binary label for each sample [0, 1, 0, ..., 1]
                If probabilistic, it is the likelihoods array of membership labels
                                  for each sample, i.e., a two-dim array [[0.6, 0.2, 0.2], ..., [..]]
        surrogates: Union[List, np.ndarray, pd.Series]
            Surrogate class attribute for each sample.
                If the membership is deterministic, surrogates are not needed.
                If the membership is probabilistic,
                    - if surrogates are given, inferred metrics are used
                                               to calculate the fairness metric as proposed in [1]_.
                    - when surrogates are not given, the arg max likelihood is used as the membership for each sample.
            Default is None.
        membership_labels: Union[int, float, str, List[int],np.array[int]]
            Labels indicating group membership.
                If the membership is deterministic, a single str/int is expected, e.g., 1.
                If the membership is probabilistic, a list or np.array of int is expected,
                                                    with the index of the protected groups in the memberships array,
                                                    e.g, [1, 2, 3], if 1-2-3 indexes are protected.
                Default value is 1 for deterministic case or [1] for probabilistic case.
        Returns
        ----------
        Pandas data frame with all implemented binary fairness metrics.
        """
        # Logic to check input types
        if is_one_dimensional(memberships):
            check_inputs(predictions, memberships, membership_labels, must_have_labels=True, labels=labels)
        elif surrogates is not None:
            check_inputs_argmax(predictions, memberships, membership_labels, labels)
        else:
            check_inputs_proba(predictions, memberships, surrogates, membership_labels, must_have_labels=True, labels=labels)

        fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]

        df = pd.DataFrame(columns=["Metric", "Value", "Ideal Value", "Lower Bound", "Upper Bound"])

        if not is_one_dimensional(memberships) and surrogates is not None:
            bootstrap_results = get_bootstrap_results(predictions, memberships, surrogates, membership_labels, labels)
        else:
            bootstrap_results = None

        for fairness_func in fairness_funcs:

            name = fairness_func[0]
            class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
            metric = class_()  # dynamically instantiate such class

            if bootstrap_results is not None and name in Constants.bootstrap_implemented:

                if membership_labels == 1:
                    membership_labels = [1]

                if name in ["PredictiveEquality", "AverageOdds", "FNRDifference"]:
                    score = metric.get_score(labels, predictions, memberships, membership_labels, bootstrap_results)
                elif name == "StatisticalParity":
                    score = metric.get_score(predictions, memberships, membership_labels, bootstrap_results)
                else:
                    score = None
            elif name in ["DisparateImpact", "StatisticalParity"]:
                score = metric.get_score(predictions, memberships, membership_labels)
            elif name in ["GeneralizedEntropyIndex", "TheilIndex"]:
                score = metric.get_score(labels, predictions)
            else:
                score = metric.get_score(labels, predictions, memberships, membership_labels)

            if score is None:
                score = np.nan
            score = np.round(score, 3)

            df = pd.concat([df,
                            pd.DataFrame([[metric.name, score, metric.ideal_value,
                                           metric.lower_bound, metric.upper_bound]], columns=df.columns)],
                           axis=0, ignore_index=True)

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

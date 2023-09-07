# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import List, Union, Optional
from typing import NamedTuple

import numpy as np
import pandas as pd

from jurity.fairness.base import _BaseBinaryFairness
from jurity.fairness.base import _BaseMultiClassMetric
from jurity.utils import is_one_dimensional
from jurity.utils_proba import get_argmax_memberships
from jurity.utils_proba import get_bootstrap_results
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
                       memberships: Union[List, np.ndarray, pd.Series],
                       surrogates: Union[List, np.ndarray, pd.Series] = None,
                       membership_labels: Union[str, float, int, List, np.array] = 1,
                       bootstrap_results: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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
        bootstrap_results: Optional[pd.DataFrame]
            A Pandas dataframe with inferred scores based surrogate class memberships.
            Default value is None.
            When given, other parameters will be discarded and bootstrap results will be used.
        Returns
        ----------
        Pandas data frame with all implemented binary fairness metrics.
        """

        # if memberships is given as likelihoods WITHOUT any surrogates, then revise it to deterministic case
        is_memberships_1d = is_one_dimensional(memberships)
        if not is_memberships_1d and surrogates is None and bootstrap_results is None:
            # Subtle point: membership_labels need to be an array when membership is 2d
            # if the user didn't specify, which defaults to 1, convert 1 -> [1] automatically
            # BUT do not overwrite membership_labels, we are still in "deterministic" mode via argmax
            # In deterministic mode, we need a single primitive label like 1
            memberships = get_argmax_memberships(memberships, [1] if membership_labels == 1 else membership_labels)
            # We now converted 2d likelihoods memberships into deterministic 1d membership, set flag to true
            is_memberships_1d = True

        # Probabilistic version
        if not is_memberships_1d or bootstrap_results is not None:
            if membership_labels == 1:
                membership_labels = [1]

            if bootstrap_results is None:
                bootstrap_results = get_bootstrap_results(predictions, memberships, surrogates,
                                                          membership_labels, labels)

        # Output df
        df = pd.DataFrame(columns=["Metric", "Value", "Ideal Value", "Lower Bound", "Upper Bound"])

        fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]
        for fairness_func in fairness_funcs:

            # Get metric
            name = fairness_func[0]
            class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
            metric = class_()  # dynamically instantiate such class

            # Get score
            score = BinaryFairnessMetrics._get_score_logic(metric, name,
                                                           labels, predictions, memberships, surrogates,
                                                           membership_labels, bootstrap_results)

            # Add score
            df = pd.concat([df,
                            pd.DataFrame([[metric.name, score, metric.ideal_value,
                                           metric.lower_bound, metric.upper_bound]], columns=df.columns)],
                           axis=0, ignore_index=True)

        df = df.set_index("Metric")

        return df

    @staticmethod
    def _get_score_logic(metric, name,
                         labels, predictions,
                         memberships, surrogates,
                         membership_labels, bootstrap_results):

        # Standard deterministic calculation
        if bootstrap_results is None:
            if name in ["DisparateImpact", "StatisticalParity"]:
                score = metric.get_score(predictions, memberships, membership_labels)
            elif name in ["GeneralizedEntropyIndex", "TheilIndex"]:
                score = metric.get_score(labels, predictions)
            else:
                score = metric.get_score(labels, predictions, memberships, membership_labels)
        else:
            if name == "StatisticalParity":
                score = metric.get_score(predictions, memberships, surrogates, membership_labels, bootstrap_results)
            elif name in ["AverageOdds", "EqualOpportunity", "FNRDifference", "PredictiveEquality"]:
                score = metric.get_score(labels, predictions, memberships, surrogates,
                                         membership_labels, bootstrap_results)
            else:
                score = None

        # pretty score
        score = np.nan if score is None else np.round(score, 3)

        return score


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

# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import List, Union

import numpy as np
import pandas as pd

from jurity.fairness.base import _BaseBinaryFairness
from jurity.utils import check_and_convert_list_types,calc_is_member
from jurity.utils import check_inputs,is_deterministic
from jurity.utils import performance_measures
from jurity.utils import split_array_based_on_membership_label
from jurity.utils_proba import get_bootstrap_results


class EqualOpportunity(_BaseBinaryFairness):

    def __init__(self):
        super().__init__("Equal Opportunity",
                         "Calculate the ratio of true positives to positive examples \
                         in the dataset, :math:`TPR = TP/P`, conditioned on a protected attribute.",
                         lower_bound=-0.2,
                         upper_bound=0.2,
                         ideal_value=0)

    @staticmethod
    def get_score(labels: Union[List, np.ndarray, pd.Series],
                  predictions: Union[List, np.ndarray, pd.Series],
                  memberships: Union[List, np.ndarray, pd.Series, List[List], pd.DataFrame],
                  surrogates: Union[List, np.ndarray, pd.Series, None] = None,
                  membership_labels: Union[str, int, List, np.array] = 1,
                  bootstrap_results: pd.DataFrame = None) -> float:
        """Calculate the ratio of true positives to positive examples in the dataset, :math:`TPR = TP/P`,
        conditioned on a protected attribute.

        Parameters
        ----------
        labels: labels: Union[List, np.ndarray, pd.Series]
        Ground truth labels for each row (0/1).
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
            Binary prediction for each sample from a binary (0/1) lack-box classifier.
        memberships: Union[List, np.ndarray, pd.Series, List[List], pd.DataFrame],
            Membership attribute for each sample.
                If deterministic, it is a binary label for each sample [0, 1, 0, .., 1]
                If probabilistic, it is the likelihoods array of membership labels for each sample. [[0.6, 0.2, 0.2], .., [..]]
        surrogates: Union[List, np.ndarray, pd.Series]
            Surrogate class attribute for each sample.
                If the membership is deterministic, surrogates are not needed.
                If the membership is probabilistic,
                    - if surrogates are given, inferred metrics are used to calculate the fairness metric as proposed in [1]_.
                    - when surrogates are not given, the arg max likelihood is considered as the membership for each sample.
            Default is None.
        membership_labels: Union[int, float, str, List[int] np.array[int]]
            Labels indicating group membership.
                If the membership is deterministic, a single str/int is expected, e.g., 1. Default is 1.
                If the membership is probabilistic, a list of int or np.array of int is expected,
                    with the positions of the protected groups in the memberships vectors (e.g, [1, 2, 3])
                Default value is 1.
        bootstrap_results: Optional[pd.DataFrame]
            A Pandas dataframe with inferred scores based surrogate class memberships.
            Default value is None.

        Returns
        ----------
        Equal opportunity difference between groups.
        """

        # Logic to check input types.
        if is_deterministic(memberships) or surrogates is None:
            check_inputs(predictions, memberships, membership_labels, must_have_labels=True, labels=labels)
            # Convert to numpy arrays
            is_member = calc_is_member(memberships, membership_labels, predictions)
            predictions = check_and_convert_list_types(predictions)
            labels = check_and_convert_list_types(labels)

            # Identify the group 1 and group 2 based on membership label
            group_2_truth, group_1_truth, group_2_group_idx, group_1_group_idx = \
                split_array_based_on_membership_label(labels, is_member, membership_labels)

            if np.unique(labels[group_1_group_idx]).shape[0] == 1 or np.unique(labels[group_2_group_idx]).shape[0] == 1:
                warnings.warn("Encountered homogeneous unary ground truth either in group 2/group 1 group. \
                Equal Opportunity will be calculated but numpy will raise division by zero.")
            elif np.unique(labels[group_1_group_idx]).shape[0] == 1 and \
                np.unique(labels[group_2_group_idx]).shape[0] == 1:
                warnings.warn("Encountered homogeneous unary ground truth in both group 1/group 2. \
                          Equal Opportunity cannot be calculated.")

            tpr_group_1 = performance_measures(labels, predictions, group_1_group_idx, group_membership=True)["TPR"]
            tpr_group_2 = performance_measures(labels, predictions, group_2_group_idx, group_membership=True)["TPR"]
        else:
            if bootstrap_results is None:
                bootstrap_results=get_bootstrap_results((predictions, memberships, surrogates, membership_labels, labels))
            tpr=bootstrap_results["TPR"]
            tpr_group_1 = tpr.loc[membership_labels]
            tpr_group_2 = tpr.loc[~(tpr.index == membership_labels)]

        return tpr_group_1 - tpr_group_2

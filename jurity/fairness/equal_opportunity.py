# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import List, Union

import numpy as np
import pandas as pd

from jurity.constants import Constants
from jurity.fairness.base import _BaseBinaryFairness
from jurity.utils import check_and_convert_list_types, check_inputs, is_one_dimensional
from jurity.utils import performance_measures
from jurity.utils import split_array_based_on_membership_label
from jurity.utils_proba import check_inputs_proba, get_argmax_memberships
from jurity.utils_proba import get_bootstrap_results, unpack_bootstrap


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
                If the membership is deterministic, a single str/int is expected, e.g., 1.
                If the membership is probabilistic, a list of int or np.array of int is expected,
                    with the index of the protected groups in the memberships vectors (e.g, [1, 2, 3])
                Default value is 1 for deterministic case or [1] for probabilistic case.
        bootstrap_results: Optional[pd.DataFrame]
            A Pandas dataframe with inferred scores based surrogate class memberships.
            Default value is None.

        Returns
        ----------
        Equal opportunity difference between groups.
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

        # Standard deterministic calculation, unless bootstrap is given
        if is_memberships_1d and bootstrap_results is None:
            check_inputs(predictions, memberships, membership_labels, must_have_labels=True, labels=labels)
            # Convert to numpy arrays
            is_member = check_and_convert_list_types(memberships)
            predictions = check_and_convert_list_types(predictions)
            labels = check_and_convert_list_types(labels)

            # Identify the group 1 and group 2 based on membership label
            group_2_truth, group_1_truth, group_2_group_idx, group_1_group_idx = \
                split_array_based_on_membership_label(labels, is_member, membership_labels)

            if np.unique(labels[group_1_group_idx]).shape[0] == 1 or np.unique(labels[group_2_group_idx]).shape[0] == 1:
                warnings.warn("Encountered homogeneous unary ground truth either in group 2/group 1 group. "
                              "Equal Opportunity will be calculated but numpy will raise division by zero.")
            elif (np.unique(labels[group_1_group_idx]).shape[0] == 1 and
                  np.unique(labels[group_2_group_idx]).shape[0] == 1):
                warnings.warn("Encountered homogeneous unary ground truth in both group 1/group 2. "
                              "Equal Opportunity cannot be calculated.")

            tpr_group_1 = performance_measures(labels, predictions, group_1_group_idx, group_membership=True)["TPR"]
            tpr_group_2 = performance_measures(labels, predictions, group_2_group_idx, group_membership=True)["TPR"]
        else:
            if membership_labels == 1:
                membership_labels = [1]

            if bootstrap_results is None:
                check_inputs_proba(predictions, memberships, surrogates, membership_labels,
                                   must_have_labels=True, labels=labels)
                bootstrap_results = get_bootstrap_results(predictions, memberships, surrogates, membership_labels,
                                                          labels)

            tpr_group_1, tpr_group_2 = unpack_bootstrap(bootstrap_results, Constants.TPR, membership_labels)

        return tpr_group_1 - tpr_group_2

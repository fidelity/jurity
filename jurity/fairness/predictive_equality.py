# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import List, Union

import numpy as np
import pandas as pd

from jurity.fairness.base import _BaseBinaryFairness
from jurity.utils import check_and_convert_list_types
from jurity.utils import check_inputs
from jurity.utils import performance_measures
from jurity.utils import split_array_based_on_membership_label


class PredictiveEquality(_BaseBinaryFairness):

    def __init__(self):
        super().__init__("Predictive Equality",
                         "We define the predictive equality as the situation when accuracy \
                         of decisions is equal across two groups, as measured by false positive rate (FPR).",
                         lower_bound=-0.2,
                         upper_bound=0.2,
                         ideal_value=0)

    @staticmethod
    def get_score(labels: Union[List, np.ndarray, pd.Series],
                  predictions: Union[List, np.ndarray, pd.Series],
                  is_member: Union[List, np.ndarray, pd.Series],
                  membership_label: Union[str, float, int] = 1) -> float:
        """
        We define the predictive equality as the situation when accuracy of decisions is equal across race groups,
        as measured by false positive rate (FPR).

        Drawing the analogy of gender classification where race is the protected attribute, across all race groups,
        the ratio of men incorrectly predicted to be a woman is the same.

        More formally,

        .. math::

            E[d(X)|Y=0, g(X)] = E[d(X), Y=0]

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
        Predictive Equality difference between groups.
        """

        # Check input types
        check_inputs(predictions, is_member, membership_label, must_have_labels=True, labels=labels)

        # Convert to numpy arrays
        is_member = check_and_convert_list_types(is_member)
        predictions = check_and_convert_list_types(predictions)
        labels = check_and_convert_list_types(labels)

        # Identify the group 1 and group 2 based on membership label
        group_2_truth, group_1_truth, group_2_group_idx, group_1_group_idx = \
            split_array_based_on_membership_label(labels, is_member, membership_label)

        if np.unique(group_2_truth).shape[0] == 1 or np.unique(group_1_truth).shape[0] == 1:
            return warnings.warn("Encountered homogeneous unary ground truth either in group 2/group 1 group. "
                                 "Predictive Equality cannot be calculated.")

        fpr_group_1 = performance_measures(labels, predictions, group_1_group_idx, group_membership=True)["FPR"]
        fpr_group_2 = performance_measures(labels, predictions, group_2_group_idx, group_membership=True)["FPR"]

        return fpr_group_1 - fpr_group_2

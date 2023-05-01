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


class AverageOdds(_BaseBinaryFairness):

    def __init__(self):
        super().__init__("Average Odds",
                         "The average odds denote the average of difference in \
                         FPR and TPR for group 1 and group 2 groups:",
                         lower_bound=-0.2,
                         upper_bound=0.2,
                         ideal_value=0)

    @staticmethod
    def get_score(labels: Union[List, np.ndarray, pd.Series],
                  predictions: Union[List, np.ndarray, pd.Series],
                  is_member: Union[List, np.ndarray, pd.Series],
                  membership_label: Union[str, float, int] = 1) -> float:
        """
        The average odds denote the average of difference in FPR and TPR for group 1 and group 2.

        .. math::
            \\frac{1}{2} [(FPR_{D = \\text{group 1}} - FPR_{D =
            \\text{group 2}}) + (TPR_{D = \\text{group 2}} - TPR_{D
            = \\text{group 1}}))]

        If predictions within ANY group are homogeneous, we cannot calculate some of the performance measures
        (such as TPR,TNR,FPR,FNR), in this case, NaN is returned.

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
        Average odds difference between groups.
        """

        # Logic to check input types
        check_inputs(predictions, is_member, membership_label, must_have_labels=True, labels=labels)

        # List needs to be converted to numpy for indexing
        is_member = check_and_convert_list_types(is_member)
        predictions = check_and_convert_list_types(predictions)
        labels = check_and_convert_list_types(labels)

        # Identify the group 2 and group 1 group based on membership label
        group_2_truth, group_1_truth, group_2_group_idx, group_1_group_idx = \
            split_array_based_on_membership_label(labels, is_member, membership_label)

        if np.unique(group_2_truth).shape[0] == 1 or np.unique(group_1_truth).shape[0] == 1:
            return warnings.warn("Encountered homogeneous unary ground truth either in group 2/group 1 group. "
                                 "Average Odds cannot be calculated.")

        results_group_1 = performance_measures(labels, predictions, group_idx=group_1_group_idx,
                                               group_membership=True)
        results_group_2 = performance_measures(labels, predictions, group_idx=group_2_group_idx,
                                               group_membership=True)

        fpr_group_1 = results_group_1["FPR"]
        fpr_group_2 = results_group_2["FPR"]
        tpr_group_1 = results_group_1["TPR"]
        tpr_group_2 = results_group_2["TPR"]

        return 0.5 * (fpr_group_1 - fpr_group_2) + 0.5 * (tpr_group_1 - tpr_group_2)

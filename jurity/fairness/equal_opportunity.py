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
                  is_member: Union[List, np.ndarray, pd.Series],
                  membership_label: Union[str, float, int] = 1) -> float:
        """Calculate the ratio of true positives to positive examples in the dataset, :math:`TPR = TP/P`,
        conditioned on a protected attribute.

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
        Equal opportunity difference between groups.
        """

        # Logic to check input types.
        check_inputs(predictions, is_member, membership_label, must_have_labels=True, labels=labels)

        # List needs to be converted to np for indexing
        is_member = check_and_convert_list_types(is_member)
        predictions = check_and_convert_list_types(predictions)
        labels = check_and_convert_list_types(labels)

        # Identify the group 2 and group 1 group based on membership label
        group_2_group_idx, group_1_group_idx, = \
            split_array_based_on_membership_label(None, is_member, membership_label, return_index_only=True)

        if np.unique(labels[group_1_group_idx]).shape[0] == 1 or np.unique(labels[group_2_group_idx]).shape[0] == 1:
            warnings.warn("Encountered homogeneous unary ground truth either in group 2/group 1 group. \
            Equal Opportunity will be calculated but numpy will raise division by zero.")
        elif np.unique(labels[group_1_group_idx]).shape[0] == 1 and \
                np.unique(labels[group_2_group_idx]).shape[0] == 1:
            warnings.warn("Encountered homogeneous unary ground truth in both group 1/group 2. \
                          Equal Opportunity cannot be calculated.")

        tpr_group_1 = performance_measures(labels, predictions, group_1_group_idx, group_membership=True)["TPR"]
        tpr_group_2 = performance_measures(labels, predictions, group_2_group_idx, group_membership=True)["TPR"]

        return tpr_group_1 - tpr_group_2

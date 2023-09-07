# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import List, Union

import numpy as np
import pandas as pd

from jurity.fairness.base import _BaseBinaryFairness, _BaseMultiClassMetric
from jurity.utils import check_and_convert_list_types
from jurity.utils import check_inputs
from jurity.utils import split_array_based_on_membership_label


class BinaryDisparateImpact(_BaseBinaryFairness):

    def __init__(self):
        super().__init__("Disparate Impact",
                         "Disparate Impact is the ratio of predictions for a 'positive' outcome in a binary "
                         "classification task between members of group 1 and group 2, respectively.",
                         lower_bound=0.8,
                         upper_bound=1.2,
                         ideal_value=1)

    @staticmethod
    def get_score(predictions: Union[List, np.ndarray, pd.DataFrame],
                  memberships: Union[List, np.ndarray, pd.DataFrame],
                  membership_label: Union[str, float, int] = 1) -> float:
        """
        Disparate Impact is the ratio of predictions for a "positive" outcome in a binary classification task
        between members of group 1 and group 2, respectively.

        .. math::

            \\frac{Pr(\\hat{Y} = 1 | D = \\text{group 1})}
                {Pr(\\hat{Y} = 1 | D = \\text{group 2})}

        Parameters
        ----------
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        memberships: Union[List, np.ndarray, pd.Series]
            Binary membership labels (0/1).
        membership_label: Union[str, float, int]
            Value indicating group membership.
            Default value is 1.

        Returns
        ----------
        Disparate impact between groups.
        """

        # Logic to check input types
        check_inputs(predictions, memberships, membership_label)

        # List needs to be converted to numpy for indexing
        memberships = check_and_convert_list_types(memberships)
        predictions = check_and_convert_list_types(predictions)

        # Identify groups based on membership label
        group_2_predictions, group_1_predictions, group_2_group, group_1_group = \
            split_array_based_on_membership_label(predictions, memberships, membership_label)

        if (group_1_predictions == 1).sum() == 0 and (group_2_predictions == 1).sum() == 0:
            warnings.warn("No positive predictions in the dataset, cannot calculate Disparate Impact.")
            return np.nan

        # Handle division by zero when no positive cases in the group 2 group
        if (group_2_predictions == 1).sum() == 0:
            warnings.warn(
                "No positive predictions found in the group 2 group. Double-check your model works correctly.")
            return (group_1_predictions == 1).sum()

        # Calculate percentages of positive predictions stratified by group membership
        group_1_predictions_pos_ratio = np.sum(group_1_predictions == 1) / len(group_1_group)
        group_2_predictions_pos_ratio = np.sum(group_2_predictions == 1) / len(group_2_group)

        return group_1_predictions_pos_ratio / group_2_predictions_pos_ratio


class MultiDisparateImpact(_BaseMultiClassMetric):

    def __init__(self, list_of_classes):
        super().__init__("Disparate Impact",
                         "Measures the ratio of predictions for a 'positive' outcome in a \
                         classification task (One vs. Others) between members of group 1 and group 2, respectively.",
                         lower_bound=0.8,
                         upper_bound=1.2,
                         ideal_value=1,
                         list_of_classes=list_of_classes)

    def _binary_score(self, predictions, is_member, membership_label=1):
        from jurity.fairness import BinaryFairnessMetrics
        return BinaryFairnessMetrics().DisparateImpact().get_score(predictions, is_member, membership_label)

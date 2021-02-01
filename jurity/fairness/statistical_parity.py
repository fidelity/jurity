# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import numpy as np
import pandas as pd

from jurity.fairness.base import _BaseBinaryFairness, _BaseMultiClassMetric
from jurity.utils import check_and_convert_list_types
from jurity.utils import check_inputs_validity
from jurity.utils import split_array_based_on_membership_label


class BinaryStatisticalParity(_BaseBinaryFairness):

    def __init__(self):
        super().__init__("Statistical Parity",
                         "Measures the difference in statistical parity between two groups",
                         lower_bound=-0.2,
                         upper_bound=0.2,
                         ideal_value=0)

    @staticmethod
    def get_score(predictions: Union[List, np.ndarray, pd.Series],
                  is_member: Union[List, np.ndarray, pd.Series],
                  membership_label: Union[str, float, int] = 1) -> float:
        """
        Difference in statistical parity between two groups.

        .. math::

            P(Y_{hat}=1 | group = \\text{group 1} ) - P(Y_{hat} = 1 | \\text{group 2})

        Parameters
        ----------
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        is_member: Union[List, np.ndarray, pd.Series]
            Binary membership labels (0/1).
        membership_label: Union[str, float, int]
            Value indicating group membership.
            Default value is 1.

        Returns
        ----------
        Statistical parity difference between groups.
        """

        # Check input types
        check_inputs_validity(predictions=predictions, is_member=is_member)

        # Convert lists to numpy arrays
        is_member = check_and_convert_list_types(is_member)
        predictions = check_and_convert_list_types(predictions)

        # Identify the group 2 and group 1 group based on specified group label
        group_2_predictions, group_1_predictions, group_2_group, group_1_group = \
            split_array_based_on_membership_label(predictions, is_member, membership_label)

        group_1_predictions_pct = np.sum(group_1_predictions == 1) / len(group_1_group)
        group_2_predictions_pct = np.sum(group_2_predictions == 1) / len(group_2_group)

        return group_1_predictions_pct - group_2_predictions_pct


class MultiStatisticalParity(_BaseMultiClassMetric):

    def __init__(self, list_of_classes):
        super().__init__("Statistical Parity",
                         "Measures the difference in statistical parity between two groups",
                         lower_bound=-0.2,
                         upper_bound=0.2,
                         ideal_value=0,
                         list_of_classes=list_of_classes)

    def _binary_score(self, predictions, is_member):
        from jurity.fairness import BinaryFairnessMetrics
        return BinaryFairnessMetrics().StatisticalParity().get_score(predictions, is_member)

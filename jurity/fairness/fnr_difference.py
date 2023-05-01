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


class FNRDifference(_BaseBinaryFairness):

    def __init__(self):
        super().__init__("FNR difference",
                         "Measures the equality (or lack thereof) of the false negative rates across groups.",
                         lower_bound=-0.2,
                         upper_bound=0.2,
                         ideal_value=0)

    @staticmethod
    def get_score(labels: Union[List, np.ndarray, pd.Series],
                  predictions: Union[List, np.ndarray, pd.Series],
                  is_member: Union[List, np.ndarray, pd.Series],
                  membership_label: Union[str, float, int] = 1) -> float:
        """
        The equality (or lack thereof) of the false negative rates across groups is an important fairness metric.
        In practice, this metric is implemented as a difference between the metric value for group 1 and group 2.

        .. math::

            E[d(X)=0 \\mid Y=1, g(X)] = E[d(X)=0, Y=1]

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
        False Negative Rate difference between groups.
        """
        # Logic to check input types.
        check_inputs(predictions, is_member, membership_label, must_have_labels=True, labels=labels)

        # List needs to be converted to np for indexing
        is_member = check_and_convert_list_types(is_member)
        predictions = check_and_convert_list_types(predictions)
        labels = check_and_convert_list_types(labels)

        # Identify the group 2 and group 1 group based on membership label
        group_2_truth, group_1_truth, group_2_group_idx, group_1_group_idx = \
            split_array_based_on_membership_label(labels, is_member, membership_label)

        if (group_2_truth == 1).sum() == 0 or (group_1_truth == 1).sum() == 0:
            warnings.warn("Encountered homogeneous unary ground truth either in group 2/group 1 group. \
                           FNR difference cannot be calculated.")
            return np.nan

        fnr_group_1 = performance_measures(labels, predictions, group_1_group_idx, group_membership=True)["FNR"]
        fnr_group_2 = performance_measures(labels, predictions, group_2_group_idx, group_membership=True)["FNR"]

        return fnr_group_1 - fnr_group_2

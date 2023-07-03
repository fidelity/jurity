# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import numpy as np
import pandas as pd

from jurity.fairness.base import _BaseBinaryFairness, _BaseMultiClassMetric
from jurity.utils import calc_is_member, check_inputs_proba, check_and_convert_list_types,Union
from jurity.utils import split_array_based_on_membership_label, is_deterministic, get_argmax_membership
from jurity.utils_proba import get_bootstrap_results


class BinaryStatisticalParity(_BaseBinaryFairness):

    def __init__(self):
        super().__init__("Statistical Parity",
                         "Measures the difference in statistical parity between two groups",
                         lower_bound=-0.2,
                         upper_bound=0.2,
                         ideal_value=0)

    @staticmethod
    def get_score(predictions: Union[List, np.ndarray, pd.Series],
                  memberships: Union[List, np.ndarray, pd.Series, List[List], pd.DataFrame],
                  surrogates: Union[List, np.ndarray, pd.Series, None] = None,
                  membership_labels: Union[str, int, List, np.array] = 1,
                  bootstrap_results: Optional[pd.DataFrame] = None) -> float:
        """
        Difference in statistical parity between two groups.

        .. math::

            P(Y_{hat}=1 | group = \\text{group 1} ) - P(Y_{hat} = 1 | \\text{group 2})

        Parameters
        ----------
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
        Statistical parity difference between groups.

        References
        ----------
            .. [1] M. Thielbar et. al., Surrogate Membership for Inferred Metrics in Fairness Evaluation, LION 2023
        """

        # Standard deterministic calculation
        if is_deterministic(memberships) or surrogates is None:
            # Check input types and determine protected class membership
            is_member=calc_is_member(memberships,membership_labels,predictions)

            predictions = check_and_convert_list_types(predictions)
            # Identify the group 2 and group 1 group based on specified group label
            group_2_predictions, group_1_predictions, group_2_group, group_1_group = \
                split_array_based_on_membership_label(predictions, is_member, membership_labels)

            group_1_predictions_pct = np.sum(group_1_predictions == 1) / len(group_1_group)
            group_2_predictions_pct = np.sum(group_2_predictions == 1) / len(group_2_group)

        # Probabilistic calculation with inferred metrics from bootstrap
        else:
            if bootstrap_results is None:
                check_inputs_proba(predictions, memberships, surrogates, membership_labels)
                bootstrap_results = get_bootstrap_results(predictions, memberships, surrogates, membership_labels)

            prediction_rate = bootstrap_results[["Prediction Rate"]]
            group_1_predictions_pct = prediction_rate.loc[membership_labels]
            group_2_predictions_pct = prediction_rate.loc[~(prediction_rate.index == membership_labels)]

        return group_1_predictions_pct - group_2_predictions_pct

class MultiStatisticalParity(_BaseMultiClassMetric):

    def __init__(self, list_of_classes):
        super().__init__("Statistical Parity",
                         "Measures the difference in statistical parity between two groups",
                         lower_bound=-0.2,
                         upper_bound=0.2,
                         ideal_value=0,
                         list_of_classes=list_of_classes)

    def _binary_score(self, predictions, is_member, membership_label=1):
        from jurity.fairness import BinaryFairnessMetrics
        return BinaryFairnessMetrics().StatisticalParity().get_score(predictions, is_member, membership_label)

# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import numpy as np
import pandas as pd

from jurity.constants import Constants
from jurity.fairness.base import _BaseBinaryFairness, _BaseMultiClassMetric
from jurity.utils import check_and_convert_list_types, Union, is_one_dimensional
from jurity.utils import split_array_based_on_membership_label, check_inputs
from jurity.utils_proba import check_inputs_proba, get_argmax_memberships
from jurity.utils_proba import get_bootstrap_results, unpack_bootstrap


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
            Binary prediction for each sample from a binary (0/1) black-box classifier.
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
            When given, other parameters will be discarded and bootstrap results will be used.

        Returns
        ----------
        Statistical parity difference between groups.

        References
        ----------
            .. [1] M. Thielbar et. al., Surrogate Membership for Inferred Metrics in Fairness Evaluation, LION 2023
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
            # Check input types and convert
            check_inputs(predictions, memberships, membership_labels)
            is_member = check_and_convert_list_types(memberships)
            predictions = check_and_convert_list_types(predictions)

            # Identify the group 2 and group 1 group based on specified group label
            group_2_predictions, group_1_predictions, group_2_group, group_1_group = \
                split_array_based_on_membership_label(predictions, is_member, membership_labels)

            group_1_predictions_pct = np.sum(group_1_predictions == 1) / len(group_1_group)
            group_2_predictions_pct = np.sum(group_2_predictions == 1) / len(group_2_group)

        # Probabilistic calculation with inferred metrics from bootstrap
        else:
            # We are in probabilistic mode, so membership labels need to be a list
            # If the user didn't specify, which defaults to 1, we can now overwrite it to [1]
            if membership_labels == 1:
                membership_labels = [1]

            if bootstrap_results is None:
                check_inputs_proba(predictions, memberships, surrogates, membership_labels)
                bootstrap_results = get_bootstrap_results(predictions, memberships, surrogates, membership_labels)

            group_1_predictions_pct, group_2_predictions_pct = unpack_bootstrap(bootstrap_results,
                                                                                Constants.PRED_RATE,
                                                                                membership_labels)
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

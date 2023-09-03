# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import List, Union

import numpy as np
import pandas as pd

from jurity.fairness.base import _BaseBinaryFairness
from jurity.utils import is_deterministic,check_and_convert_list_types, check_inputs
from jurity.utils import calc_is_member
from jurity.utils import performance_measures
from jurity.utils import split_array_based_on_membership_label
from jurity.utils_proba import get_bootstrap_results,unpack_bootstrap

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
                  memberships: Union[List, np.ndarray, pd.Series, List[List], pd.DataFrame],
                  surrogates: Union[List, np.ndarray, pd.Series, None] = None,
                  membership_labels: Union[str, int, List, np.array] = 1,
                  bootstrap_results: pd.DataFrame = None) -> float:
        """
        The equality (or lack thereof) of the false negative rates across groups is an important fairness metric.
        In practice, this metric is implemented as a difference between the metric value for group 1 and group 2.

        .. math::

            E[d(X)=0 \\mid Y=1, g(X)] = E[d(X)=0, Y=1]

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
        False Negative Rate difference between groups.
        """
        # Logic to check input types.
        if is_deterministic(memberships) or (surrogates is None and bootstrap_results is None):
            check_inputs(predictions, memberships, membership_labels, must_have_labels=True, labels=labels)
            # Convert to numpy arrays
            is_member = calc_is_member(memberships, membership_labels, predictions)
            predictions = check_and_convert_list_types(predictions)
            labels = check_and_convert_list_types(labels)

            # Identify the group 1 and group 2 based on membership label
            group_2_truth, group_1_truth, group_2_group_idx, group_1_group_idx = \
                split_array_based_on_membership_label(labels, is_member, membership_labels)

            if (group_2_truth == 1).sum() == 0 or (group_1_truth == 1).sum() == 0:
                warnings.warn("Encountered homogeneous unary ground truth either in group 2/group 1 group. \
                           FNR difference cannot be calculated.")
                return np.nan
            fnr_group_1 = performance_measures(labels, predictions, group_1_group_idx, group_membership=True)["FNR"]
            fnr_group_2 = performance_measures(labels, predictions, group_2_group_idx, group_membership=True)["FNR"]
        else:
            if bootstrap_results is None:
                bootstrap_results=get_bootstrap_results(predictions, memberships, surrogates, membership_labels, labels)
            fnr_group_1,fnr_group_2 = unpack_bootstrap(bootstrap_results,"FNR",membership_labels)

        return fnr_group_1 - fnr_group_2

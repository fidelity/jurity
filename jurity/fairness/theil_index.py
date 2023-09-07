# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import numpy as np
import pandas as pd

from jurity.fairness.base import _BaseBinaryFairness


class TheilIndex(_BaseBinaryFairness):

    def __init__(self):
        super().__init__("Theil Index",
                         "The Theil index is the generalized entropy index with  $alpha = 1$. \
                         See Generalized Entropy index.",
                         lower_bound=0.0,
                         upper_bound=np.inf,
                         ideal_value=0)

    def get_score(self,
                  labels: Union[List, np.ndarray, pd.Series],
                  predictions: Union[List, np.ndarray, pd.Series],
                  membership_label: Union[str, float, int] = 1) -> float:
        """
        The Theil index is the generalized entropy index with :math:`\\alpha = 1`.
        See Generalized Entropy index.

        Parameters
        ----------
        labels: Union[List, np.ndarray, pd.Series]
            Binary ground truth labels for the provided dataset (0/1).
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        membership_label: Union[str, float, int]
            Value indicating group membership.
            Default value is 1.

        Returns
        ----------
        Theil Index of the classifier.
        """
        from jurity.fairness import BinaryFairnessMetrics
        metric = BinaryFairnessMetrics.GeneralizedEntropyIndex()
        return metric.get_score(labels, predictions, membership_label, alpha=1)

# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import numpy as np
import pandas as pd

from jurity.fairness.base import _BaseBinaryFairness
from jurity.utils import check_and_convert_list_types, check_binary, check_elementwise_input_type
from jurity.utils import check_input_shape, check_input_type
from jurity.utils import check_true


class TheilIndex(_BaseBinaryFairness):

    def __init__(self, positive_label_name: float = 1):
        super().__init__("Theil Index",
                         "The Theil index is the generalized entropy index with  $alpha = 1$. \
                         See Generalized Entropy index.",
                         lower_bound=0.0,
                         upper_bound=np.inf,
                         ideal_value=0)

        self.positive_label_name = positive_label_name

    def get_score(self, labels: Union[List, np.ndarray, pd.Series],
                  predictions: Union[List, np.ndarray, pd.Series]) -> float:
        """
        The Theil index is the generalized entropy index with :math:`\\alpha = 1`.
        See Generalized Entropy index.

        Parameters
        ----------
        labels: Union[List, np.ndarray, pd.Series]
            Binary ground truth labels for the provided dataset (0/1).
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).

        Returns
        ----------
        Theil Index of the classifier.
        """
        check_input_type(labels)
        check_input_type(predictions)

        # Check input shape
        check_input_shape(labels)
        check_input_shape(predictions)

        # Check input content
        check_binary(labels)
        check_binary(predictions)

        # Check the actual contents of the arrays
        check_elementwise_input_type(labels)
        check_elementwise_input_type(predictions)

        # Check that our arrays are all the same length
        check_true(len(labels) == len(predictions),
                   AssertionError("Shapes of inputs do not match. You supplied labels :"
                                  f"{len(labels)} and predictions: {len(predictions)}"))

        # Convert lists
        y_pred = check_and_convert_list_types(predictions)
        y_true = check_and_convert_list_types(labels)

        y_pred = (y_pred == self.positive_label_name).astype(np.float64)
        y_true = (y_true == self.positive_label_name).astype(np.float64)

        b = 1 + y_pred - y_true

        return float(np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b)))

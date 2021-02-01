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


class GeneralizedEntropyIndex(_BaseBinaryFairness):

    def __init__(self, positive_label_name: float = 1):
        super().__init__("Generalized Entropy Index",
                         "Generalized entropy index is proposed as a unified individual and group fairness measure.",
                         lower_bound=0.0,
                         upper_bound=np.inf,
                         ideal_value=0)

        self.positive_label_name = positive_label_name

    def get_score(self,
                  labels: Union[List, np.ndarray, pd.Series],
                  predictions: Union[List, np.ndarray, pd.Series],
                  alpha: float = 2) -> float:
        """Generalized entropy index is proposed as a unified individual and group fairness measure in [3]_.
        With :math:`b_i = \\hat{y}_i - y_i + 1`:

        .. math::

           \\mathcal{E}(\\alpha) = \\begin{cases}
              \\frac{1}{n \\alpha (\\alpha-1)}\\sum_{i=1}^n\\left[\\left(\\frac{b_i}{\\mu}\\right)^\\alpha - 1\\right] &
              \\alpha \\ne 0, 1, \\\\
              \\frac{1}{n}\\sum_{i=1}^n\\frac{b_{i}}{\\mu}\\ln\\frac{b_{i}}{\\mu} & \\alpha=1, \\\\
            -\\frac{1}{n}\\sum_{i=1}^n\\ln\\frac{b_{i}}{\\mu},& \\alpha=0.
            \\end{cases}

        Parameters
        ----------
        labels: Union[List, np.ndarray, pd.Series]
            Binary ground truth labels for the provided dataset (0/1).
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        alpha: float
            Parameter that regulates weight given to distances between values at different parts of the distribution.
            Default value is 2.

        Returns
        ----------
        General Entropy Index of the classifier.

        References:
        ----------
            .. [3] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
             A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via
             Inequality Indices, ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.
        """

        # Check input types
        check_input_type(labels)
        check_input_type(predictions)

        # Check input shapes
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

        # Convert
        y_pred = check_and_convert_list_types(predictions)
        y_true = check_and_convert_list_types(labels)

        y_pred = (y_pred == self.positive_label_name).astype(np.float64)
        y_true = (y_true == self.positive_label_name).astype(np.float64)

        b = 1 + y_pred - y_true

        if alpha == 1:
            # moving the b inside the log allows for 0 values
            return float(np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b)))
        elif alpha == 0:
            return -np.mean(np.log(b / np.mean(b)))
        else:
            return np.mean((b / np.mean(b)) ** alpha - 1) / (alpha * (alpha - 1))

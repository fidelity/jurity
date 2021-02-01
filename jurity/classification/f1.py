# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from jurity.utils import check_binary_values


class F1:

    @staticmethod
    def get_score(actual: Union[List, np.ndarray, pd.Series],
                  predicted: Union[List, np.ndarray, pd.Series],
                  sample_weight: Optional[Union[List, np.ndarray, pd.Series]] = None) -> float:
        """
        Compute the F1 score, also known as balanced F-score or F-measure.

        The F1 score is a weighted average of precision and recall, with equal relative contribution.
        The best value is 1 and the worst value is 0.

        The formula for the F1 score is::
            F1 = 2 * (precision * recall) / (precision + recall)

        Parameters
        ----------
        actual: Union[List, np.ndarray, pd.Series]
            Binary ground truth (correct) labels (0/1).
        predicted: Union[List, np.ndarray, pd.Series]
            Binary predicted labels, as returned by a classifier (0/1).
        sample_weight: Union[List, np.ndarray, pd.Series]
            Sample weights.

        Returns
        -------
        Recall score.
        """

        # Check input
        check_binary_values(actual)
        check_binary_values(predicted)

        return f1_score(actual, predicted, pos_label=1, average="binary", sample_weight=sample_weight)

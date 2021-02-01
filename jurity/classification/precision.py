# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

from jurity.utils import check_binary_values


class Precision:

    @staticmethod
    def get_score(actual: Union[List, np.ndarray, pd.Series],
                  predicted: Union[List, np.ndarray, pd.Series],
                  sample_weight: Optional[Union[List, np.ndarray, pd.Series]] = None) -> float:
        """
        Calculates precision.

        The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of true positives and ``fp`` the number
        of false positives. The precision is intuitively the ability of the classifier not to label as positive a
        sample that is negative.

        The best value is 1 and the worst value is 0.

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
        Precision score.
        """

        # Check input
        check_binary_values(actual)
        check_binary_values(predicted)

        return precision_score(actual, predicted, pos_label=1, average="binary", sample_weight=sample_weight)

# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from jurity.utils import check_binary_values


class Accuracy:

    @staticmethod
    def get_score(actual: Union[List, np.ndarray, pd.Series],
                  predicted: Union[List, np.ndarray, pd.Series],
                  sample_weight: Optional[Union[List, np.ndarray, pd.Series]] = None) -> float:
        """
        Calculates accuracy score as the fraction of correctly classified samples.

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
        Accuracy score.
        """

        # Check input
        check_binary_values(actual)
        check_binary_values(predicted)

        return accuracy_score(actual, predicted, normalize=True, sample_weight=sample_weight)

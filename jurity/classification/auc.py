# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jurity.utils import check_binary_values, check_likelihood_values


class AUC:

    @staticmethod
    def get_score(actual: Union[List, np.ndarray, pd.Series],
                  likelihoods: Union[List, np.ndarray, pd.Series],
                  sample_weight: Optional[Union[List, np.ndarray, pd.Series]] = None) -> float:
        """
        Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from predicted likelihoods.

        Parameters
        ----------
        actual: Union[List, np.ndarray, pd.Series]
            Binary ground truth (correct) labels (0/1).
        likelihoods: Union[List, np.ndarray, pd.Series]
            Predicted likelihoods, as returned by a classifier.
        sample_weight: Union[List, np.ndarray, pd.Series]
            Sample weights.

        Returns
        -------
        Recall score.
        """

        # Check input
        check_binary_values(actual)
        check_likelihood_values(likelihoods)

        return roc_auc_score(actual, likelihoods, sample_weight=sample_weight)

from typing import NamedTuple

import numpy as np
import pandas as pd

from jurity.utils import Constants, get_sorted_clicks


class RankEstimation(NamedTuple):
    """ Unbiased Rank Estimation Wrapper

    Parameters
    ----------
        n_items: int
            The total number of items available for recommendation.
        n_sampled: int
            The number of items sampled for the response matrix.

    """
    n_items: int = 1
    n_sampled: int = 1


class UnbiasedRankEstimation:
    def __init__(self, rank_estimation: RankEstimation):
        """ Unbiased Rank Estimation

        Performs a correction of ranking metrics for use cases where the negative responses were sampled.
        This adjusts the metric for when if the full set of items were used,
        items in the sampled top k would not be ranked as high.

        .. math::
            URE = metric \\times (1 + \\frac{(n_sampled -1)(predicted_rank - 1)}{n_items})

        Parameters
        ----------
        rank_estimation: NamedTuple
            The RankEstimation named tuple with n_items and n_sampled.
        """

        self.n_items = rank_estimation.n_items
        self.n_sampled = rank_estimation.n_sampled

    def get_correction(self, predicted_results: pd.DataFrame):
        """
        Calculates the correction value for the given predictions.

        Parameters
        ----------
        predicted_results: pd.DataFrame
            A pandas DataFrame for the recommended user item interaction data, captured from a recommendation algorithm.
            The DataFrame should contain a minimum of two columns, including self._user_id_column, self._item_id_column,
            and anything else the metric may need. Each row contains the interaction of one user with one item, and the
            scores associated with this interaction. There can be multiple interactions per user, and there can be
            multiple users per DataFrame. However, the interactions for a specific user must be contained within a
            single DataFrame.

        """
        pass


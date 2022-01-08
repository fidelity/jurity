# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import warnings

from jurity.utils import Constants


def sample_users(df: pd.DataFrame, user_id_column: str = Constants.user_id,
                 sample_size: float = None, seed: int = Constants.default_seed) -> pd.DataFrame:
    """
    Samples input data frame by selecting a random sample of users.

    Parameters
    ----------
    df: pd.DataFrame
        Data frame with a user_id_col column.
    user_id_column: str
        User id column name.
    sample_size: float
        Proportion of users to randomly sample for evaluation.
    seed : int, default=Constants.default_seed
        The seed used to create random state.

    Returns
    -------
    Sampled data frame
    """
    rng = np.random.RandomState(seed)
    users = df[user_id_column].unique()
    users = rng.choice(users, size=int(len(users) * sample_size), replace=False)
    return df[df[user_id_column].isin(users)]


def tocsr(df: pd.DataFrame, user_id_column: str = Constants.user_id, item_id_column: str = Constants.item_id):
    """
    Transform data frame with user_id and item_id columns to sparse csr matrix.

    Parameters
    ----------
    df: pd.DataFrame
        Data frame with (user_id, item_id) in each row.
    user_id_column: str
        User id column name.
    item_id_column: str
        Item id column name.

    Returns
    -------
    Sparse matrix with user-item interactions/recommendations.
    """

    # Map each user_id to (0, n_users)
    users = list(df[user_id_column].unique())
    user_map = dict(zip(users, range(len(users))))

    # Map each user_id to (0, n_items)
    items = list(df[item_id_column].unique())
    item_map = dict(zip(items, range(len(items))))

    # Update user_id, item_id values
    user_ids = df[user_id_column].map(user_map).values
    item_ids = df[item_id_column].map(item_map).values

    # Convert to csr matrix
    csr_matrix = sp.coo_matrix((np.ones(len(user_ids)), (user_ids, item_ids))).tocsr()

    return csr_matrix


def interlist_diversity(predicted_results: pd.DataFrame, click_column: str, k: int,
                        user_id_column: str = Constants.user_id, item_id_column: str = Constants.item_id,
                        sample_size: float = None, seed: int = Constants.default_seed,
                        chunk_size: int = 1000) -> Tuple[float, int]:
    """
        Calculate inter-list diversity metric:

            Inter-list-diversity = 1 - average(cosine_similarity(R_{u_i}, R_{u_j})),
            where R_{u_i} is the binary indicator vector representing provided recommendations for user u_i, i \ne j.

        Parameters
        ----------
        predicted_results: pd.DataFrame
            Recommendations data frame with (user_id, item_id, score) in each row.
        k: int
            Top-k recommendations to consider.
        user_id_column: str
            User id column name.
        item_id_column: str
            Item id column name.
        click_column: str
            Recommendation score column name.
        sample_size: float
            Proportion of users to randomly sample for evaluation.
        seed : int
            The seed used to create random state.
        chunk_size: int
            Chunk size to limit memory usage.

        Returns
        -------
        Inter-list diversity metric, number of unique users as the support to get the metric
        """

    # Sample users
    if sample_size is not None:
        df = sample_users(predicted_results, user_id_column, sample_size, seed)
    else:
        df = predicted_results

    # Sort by user and score
    df = df.sort_values(click_column, ascending=False).groupby(user_id_column).head(k)

    # Given user/item id maps, create sparse matrix.
    sparse_matrix = tocsr(df, user_id_column, item_id_column)

    # Get pairwise cosine similarities
    similarities_sum = 0
    if chunk_size == 1:
        num_chunks = sparse_matrix.shape[0]
    else:
        num_chunks = sparse_matrix.shape[0] // chunk_size + 1
    for i in range(num_chunks):
        similarities = cosine_similarity(sparse_matrix[i * chunk_size:(i + 1) * chunk_size], sparse_matrix,
                                         dense_output=False)
        similarities_sum += similarities.sum()
    similarities_sum = (similarities_sum - sparse_matrix.shape[0]) / 2.0

    # Get number of pairs
    num_pairs = np.sum(range(sparse_matrix.shape[0]))

    # Calculate metric
    if num_pairs == 0:
        inter_list_diversity = np.nan
        warnings.warn('Inter-List Diversity will be nan when there is only one single user.')
    else:
        inter_list_diversity = 1.0 - similarities_sum / num_pairs
        if np.abs(inter_list_diversity) <= 1e-06:
            inter_list_diversity = 0.0

    # Calculate support, set it to be the number of users
    support = len(df[user_id_column].unique())

    return inter_list_diversity, support


class InterListDiversity:
    """Inter-List Diversity@k

    Inter-List Diversity@k measures the inter-list diversity of the recommendations when only k recommendations are
    made to the user. It measures how user's lists of recommendations are different from each other.

    .. math::
            Inter-list-diversity = 1 - average(cosine_similarity(R_{u_i}, R_{u_j})),
            where R_{u_i} is the binary indicator vector representing provided recommendations for user u_i, i != j.

    Sources: https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093
    """

    def __init__(self, click_column, k: int = None, user_id_column: str = Constants.user_id,
                 item_id_column: str = Constants.item_id, sample_size: float = None,
                 seed: int = Constants.default_seed, chunk_size: int = 1000):
        self.user_id_column = user_id_column
        self.item_id_column = item_id_column
        self.click_column = click_column
        self.k = k
        self.sample_size = sample_size
        self.seed = seed
        self.chunk_size = chunk_size

    def get_score(self, predicted_results: pd.DataFrame, return_extended_results: bool = False) -> Union[float, dict]:
        """Evaluates the current metric on the given data.

        Parameters
        ---------
        predicted_results: pd.DataFrame
            A pandas DataFrame for the recommended user item interaction data, captured from a recommendation algorithm.
            The DataFrame should contain a minimum of two columns, including self._user_id_column, self._item_id_column,
            and anything else the metric may need. Each row contains the interaction of one user with one item, and the
            scores associated with this interaction. There can be multiple interactions per user, and there can be
            multiple users per DataFrame. However, the interactions for a specific user must be contained within a
            single DataFrame.
        return_extended_results: bool
            Whether the extended results such as the support should also be returned. If specified, the returned results
            will be of type ``dict``. Inter-list diversity currently returns ``Inter-List Diversity`` and
            the ``support``, which is the number of unique users to calculate it.

        Returns
        -------
        metric: Union[float, dict]
            The averaged result(s). The return type is determined by ``return_extended_results`` parameters.
        """
        results, support = interlist_diversity(predicted_results, self.click_column, self.k,
                                               user_id_column=self.user_id_column,
                                               item_id_column=self.item_id_column,
                                               sample_size=self.sample_size, seed=self.seed,
                                               chunk_size=self.chunk_size)

        if return_extended_results:
            return {'inter-list diversity': results, 'support': support}
        else:
            return results

    def __str__(self):
        return 'Inter-List Diversity@{}'.format(self.k)

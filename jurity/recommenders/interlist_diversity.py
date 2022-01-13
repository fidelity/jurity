# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_chunked

import warnings

from jurity.utils import Constants, tocsr, sample_users, get_sorted_clicks, check_true


def interlist_diversity(predicted_results: pd.DataFrame, click_column: str, k: int,
                        user_id_column: str = Constants.user_id, item_id_column: str = Constants.item_id,
                        user_sample_size: Union[int, float, None] = 10000, seed: int = Constants.default_seed,
                        metric: Union[str, Callable] = 'cosine', num_runs: int = 10, n_jobs: int = 1,
                        working_memory: int = None) -> Tuple[float, int]:
    """
    Inter-List Diversity@k measures the inter-list diversity of the recommendations when only k recommendations are
    made to the user. It measures how user's lists of recommendations are different from each other. This metric has a
    range in :math:`[0, 1]`. The higher this metric is, the more diversified lists of items are recommended to different
    users. Let :math:`U` denote the set of :math:`N` unique users, :math:`u_i`, :math:`u_j \in U` denote the i-th and
    j-th user in the user set, :math:`i, j \in \{0,1,\cdots,N\}`. :math:`R_{u_i}` is the binary indicator vector
    representing provided recommendations for :math:`u_i`. :math:`I` is the set of all unique user pairs,
    :math:`\\forall~i<j, \{u_i, u_j\} \in I`.

    .. math::
            Inter \mbox{-} list~diversity = \\frac{\sum_{i,j, \{u_i, u_j\} \in I}(cosine\_distance(R_{u_i}, R_{u_j}))}{|I|}

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
    user_sample_size: Union[int, float, None]
        When input is an integer, it defines the number of randomly sampled users. When input is float, it defines the
        proportion of users to randomly sample for evaluation. If it is None, all users are included. Default=10,000.
    seed: int
        The seed used to create random state.
    metric: Union[str, Callable]
        Default = 'cosine'. The distance metric leveraged by sklearn.metrics.pairwise_distances_chunked.
        The metric to use when calculating distance between instances in a feature array.
        If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric
        parameter, or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is a callable function,
        it is called on each pair of instances (rows) and the resulting value recorded.
        The callable should take two arrays from X as input and return a value indicating the distance between them.
    num_runs: int
        num_runs is used to report the approximation of Inter-List Diversity over multiple runs on smaller
        samples of users, default=10, for a speed-up on evaluations. The sampling size is defined by
        user_sample_size. The final result is averaged over the multiple runs.
    n_jobs: int
        Number of jobs to use for computation in parallel, leveraged by sklearn.metrics.pairwise_distances_chunked.
        -1 means using all processors. Default=1.
    working_memory: Union[int, None]
        Maximum memory for temporary distance matrix chunks, leveraged by sklearn.metrics.pairwise_distances_chunked.
        When None (default), the value of sklearn.get_config()['working_memory'], i.e. 1024M, is used.

    Returns
    -------
    Inter-list diversity metric, number of unique users as the support to get the metric
    """

    # Sample users
    if user_sample_size is not None:
        results_over_runs = []
        supports_over_runs = []

        # Create a different seed for each run
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, num_runs*10, num_runs)

        for i in range(num_runs):

            df = sample_users(predicted_results, user_id_column, user_sample_size, seed=seeds[i])

            res, support = interlist_diversity(df, click_column, k, user_id_column=user_id_column,
                                               item_id_column=item_id_column, user_sample_size=None,
                                               metric=metric, n_jobs=n_jobs, working_memory=working_memory)
            results_over_runs.append(res)
            supports_over_runs.append(support)

        inter_list_diversity = np.mean(results_over_runs)
        support = int(np.mean(supports_over_runs))

        return inter_list_diversity, support

    df = predicted_results

    # Sort by user and score, and take the top k scores.
    df = get_sorted_clicks(df, user_id_column, click_column, k)

    # Given user/item id column names, create sparse matrix as the new representation of user-item interactions.
    sparse_matrix = tocsr(df, user_id_column, item_id_column)

    # Get pairwise cosine distances
    chunked_sum_cosine_distances = map(sum, pairwise_distances_chunked(sparse_matrix, reduce_func=reduce_func,
                                                                       metric=metric, n_jobs=n_jobs,
                                                                       working_memory=working_memory))

    # Sum of all cosine distances of unique pairs
    sum_cosine_distances = sum(list(chunked_sum_cosine_distances)) / 2.0

    # Get number of pairs
    num_pairs = np.sum(range(sparse_matrix.shape[0]))

    # Calculate metric
    if num_pairs == 0:
        inter_list_diversity = np.nan
        warnings.warn('Inter-List Diversity will be nan when there is only one single user.')
    else:
        inter_list_diversity = sum_cosine_distances / num_pairs
        if np.abs(inter_list_diversity) <= 1e-06:
            inter_list_diversity = 0.0

    # Calculate support, set it to be the number of users
    support = len(df[user_id_column].unique())

    return inter_list_diversity, support


def reduce_func(D_chunk, start):
    """
    The function which is applied on each chunk of the distance matrix, reducing it to needed values.
    reduce_func(D_chunk, start) is called repeatedly, where D_chunk is a contiguous vertical slice of the pairwise
    distance matrix, starting at row start. It should return one of: None; an array, a list, or a sparse matrix of
    length D_chunk.shape[0]; or a tuple of such objects. Returning None is useful for in-place operations,
    rather than reductions.
    """
    return np.sum(D_chunk, axis=1)


class InterListDiversity:
    """Inter-List Diversity@k

    Inter-List Diversity@k measures the inter-list diversity of the recommendations when only k recommendations are
    made to the user. It measures how user's lists of recommendations are different from each other. This metric has a
    range in :math:`[0, 1]`. The higher this metric is, the more diversified lists of items are recommended to different
    users. Let :math:`U` denote the set of :math:`N` unique users, :math:`u_i`, :math:`u_j \in U` denote the i-th and
    j-th user in the user set, :math:`i, j \in \{0,1,\cdots,N\}`. :math:`R_{u_i}` is the binary indicator vector
    representing provided recommendations for :math:`u_i`. :math:`I` is the set of all unique user pairs,
    :math:`\\forall~i<j, \{u_i, u_j\} \in I`.

    .. math::
            Inter \mbox{-} list~diversity = \\frac{\sum_{i,j, \{u_i, u_j\} \in I}(cosine\_distance(R_{u_i}, R_{u_j}))}{|I|}
    """

    def __init__(self, click_column, k: int = None, user_id_column: str = Constants.user_id,
                 item_id_column: str = Constants.item_id, user_sample_size: Union[int, float] = 10000,
                 seed: int = Constants.default_seed, metric: Union[str, Callable] = 'cosine', num_runs: int = 10,
                 n_jobs: int = 1, working_memory: int = None):
        """Initialize the parameters for Inter-List Diversity metric.

        Parameters
        ----------
        click_column: str
            Recommendation score column name.
        k: int
            Top-k recommendations to consider.
        user_id_column: str
            User id column name.
        item_id_column: str
            Item id column name.
        user_sample_size: Union[int, float, None]
            When input is an integer, it defines the number of randomly sampled users. When input is float, it defines
            the proportion of users to randomly sample for evaluation. If it is None, all users are included.
            Default=10,000.
        seed: int
            The seed used to create random state.
        metric: Union[str, Callable]
            Default = 'cosine'. The distance metric leveraged by sklearn.metrics.pairwise_distances_chunked.
            The metric to use when calculating distance between instances in a feature array.
            If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric
            parameter, or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is a callable function,
            it is called on each pair of instances (rows) and the resulting value recorded.
            The callable should take two arrays from X as input and return a value indicating the distance between them.
        num_runs: int
            num_runs is used to report the approximation of Inter-List Diversity over multiple runs on smaller
            samples of users, default=10, for a speed-up on evaluations. The sampling size is defined by
            user_sample_size. The final result is averaged over the multiple runs.
        n_jobs: int
            Number of jobs to use for computation in parallel, leveraged by sklearn.metrics.pairwise_distances_chunked.
            -1 means using all processors. Default=1.
        working_memory: Union[int, None]
            Maximum memory for temporary distance matrix chunks, leveraged by sklearn.metrics.pairwise_distances_chunked.
            When None (default), the value of sklearn.get_config()['working_memory'], i.e. 1024M, is used.
        """

        self.user_id_column = user_id_column
        self.item_id_column = item_id_column
        self.click_column = click_column
        self.k = k
        self.user_sample_size = user_sample_size
        self.seed = seed
        self.metric = metric
        self.num_runs = num_runs
        self.n_jobs = n_jobs
        self.working_memory = working_memory

        self._validate_arguments()

    def get_score(self, actual_results: pd.DataFrame, predicted_results: pd.DataFrame, batch_accumulate: bool = False,
                  return_extended_results: bool = False) -> Union[float, dict]:
        """Evaluates the current metric on the given data.

        Parameters
        ---------
        actual_results: Ignored.
            Ignored for calculating Inter-List Diversity while it is kept for making the API design consistent across
            different recommender metrics.
        predicted_results: pd.DataFrame
            A pandas DataFrame for the recommended user item interaction data, captured from a recommendation algorithm.
            The DataFrame should contain a minimum of two columns, including self._user_id_column, self._item_id_column,
            and anything else the metric may need. Each row contains the interaction of one user with one item, and the
            scores associated with this interaction. There can be multiple interactions per user, and there can be
            multiple users per DataFrame. However, the interactions for a specific user must be contained within a
            single DataFrame.
        batch_accumulate: bool
            Should not be True for calculating Inter-List Diversity while it is kept for making the API design
            consistent across different recommender metrics.
        return_extended_results: bool
            Whether the extended results such as the support should also be returned. If specified, the returned results
            will be of type ``dict``. Inter-list diversity currently returns ``Inter-List Diversity`` and
            the ``support``, which is the number of unique users to calculate it.

        Returns
        -------
        metric: Union[float, dict]
            The averaged result(s). The return type is determined by ``return_extended_results`` parameters.
        """

        if batch_accumulate:
            raise ValueError("Batch_accumulate can not be set as True for Inter-List Diversity.")

        results, support = interlist_diversity(predicted_results, self.click_column, self.k,
                                               user_id_column=self.user_id_column,
                                               item_id_column=self.item_id_column,
                                               user_sample_size=self.user_sample_size, seed=self.seed,
                                               metric=self.metric, n_jobs=self.n_jobs, num_runs=self.num_runs,
                                               working_memory=self.working_memory
                                               )

        if return_extended_results:
            return {'inter-list diversity': results, 'support': support}
        else:
            return results

    def _validate_arguments(self):
        """Validate arguments for Inter-List Diversity"""

        check_true(isinstance(self.num_runs, int), ValueError("num_runs should be an integer."))
        if self.user_sample_size:
            check_true(isinstance(self.user_sample_size, int) or isinstance(self.user_sample_size, float),
                       ValueError("user_sample_size should be an integer or a float number."))
            check_true(self.num_runs >= 1, ValueError("num_runs should be no less than 1."))
            check_true(isinstance(self.num_runs, int), ValueError("num_runs should be an integer."))
        check_true(isinstance(self.click_column, str), ValueError("click_column should be a string."))
        if self.k:
            check_true(isinstance(self.k, int), ValueError("k should be an integer."))
        if isinstance(self.user_sample_size, int):
            check_true(self.user_sample_size >= 1, ValueError("user_sample_size should be no less than 1."))
        elif isinstance(self.user_sample_size, float):
            check_true(self.user_sample_size > 0.0, ValueError("user_sample_size should be greater than 0.0."))
        check_true(isinstance(self.n_jobs, int), ValueError("n_jobs should be an integer."))
        if self.working_memory:
            check_true(isinstance(self.working_memory, int), ValueError("working_memory should be an integer."))

    def __str__(self):
        return 'Inter-List Diversity@{}'.format(self.k)

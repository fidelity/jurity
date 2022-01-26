# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import warnings
from jurity.utils import Constants, tocsr, sample_users, get_sorted_clicks, check_true


def intralist_diversity(predicted_results: pd.DataFrame, item_features: pd.DataFrame, click_column: str, k: int,
                        user_id_column: str = Constants.user_id, item_id_column: str = Constants.item_id,
                        user_sample_size: Union[int, float, None] = 10000, seed: int = Constants.default_seed,
                        metric: Union[str, Callable] = 'cosine', num_runs: int = 10):
    """
    Intra-List Diversity@k measures the intra-list diversity of the recommendations when only k recommendations are
    made to the user. It measures how items from the same user are different from each other. This metric has a
    range in :math:`[0, 1]`. The higher this metric is, the more diversified lists of items are recommended to different
    users. Let :math:`U` denote the set of :math:`N` unique users, :math:`u_i`, :math:`u_j \in U` denote the i-th and
    j-th user in the user set, :math:`i, j \in \{0,1,\cdots,N\}`. :math:`R_{u_i}` is the binary indicator vector
    representing provided recommendations for :math:`u_i`. :math:`I` is the set of all unique user pairs,
    :math:`\\forall~i<j, \{u_i, u_j\} \in I`.

    .. math::
            Intra \mbox{-} list~diversity = 1 - \\frac{1}{U}\sum_{i=1}^U average(consine\_similarity(v_p^{u_i}, v_q^{u_i})),

    Parameters
    ----------
    predicted_results: pd.DataFrame
        Recommendations data frame with (user_id, item_id, score) in each row.
    item_features: pd.DataFrame
        features data frame with (item_id, feature_1, feature_2, ..., feature_n) in each row.
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
        num_runs is used to report the approximation of Intra-List Diversity over multiple runs on smaller
        samples of users, default=10, for a speed-up on evaluations. The sampling size is defined by
        user_sample_size. The final result is averaged over the multiple runs.

    Returns
    -------
    Intra-list diversity metric, number of unique users as the support to get the metric
    """

    # Sample users
    if user_sample_size is not None:
        results_over_runs = []
        supports_over_runs = []

        # Create a different seed for each run
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, num_runs * 10, num_runs)

        for i in range(num_runs):
            df = sample_users(predicted_results, user_id_column, user_sample_size, seed=seeds[i])

            res, support = intralist_diversity(df, item_features, click_column, k, user_id_column=user_id_column,
                                               item_id_column=item_id_column, user_sample_size=None,
                                               metric=metric)
            results_over_runs.append(res)
            supports_over_runs.append(support)

        intra_list_diversity = np.mean(results_over_runs)
        support = int(np.mean(supports_over_runs))

        return intra_list_diversity, support

    if k == 1:
        warnings.warn('Intra-List Diversity will be nan when only one item is provided in item lists.')

    df = predicted_results

    # Sort by user and score, and take the top k scores.
    df = get_sorted_clicks(df, user_id_column, click_column, k)

    # mapping user_ids and item_ids to indices
    unique_user_ids = list(df[user_id_column].unique())
    user_id_map = dict(zip(unique_user_ids, range(len(unique_user_ids))))

    item_features = item_features.reset_index(drop=True)
    item_id_map = {v: k for k, v in item_features[item_id_column].to_dict().items()}

    df['item_index'] = df[item_id_column].map(item_id_map).values
    df['user_index'] = df[user_id_column].map(user_id_map).values

    # cross join on user_id to get all 2-item combinations of items in lists
    df_merged = df.merge(df, on=user_id_column)

    # remove item pairs with the same items
    df_merged = df_merged[df_merged['item_index_x'] != df_merged['item_index_y']][
        [user_id_column, 'item_index_x', 'item_index_y']]

    ## recompute pairwise distances for all combinations of items in the features dataframe
    item_feature_minus = item_features.set_index(item_id_column)
    cosine_distance = pairwise_distances(item_feature_minus, metric=metric)

    # fetch distance for every pair in every item pair
    df_merged['cosine_distance'] = cosine_distance[df_merged['item_index_x'], df_merged['item_index_y']]
    results = df_merged[[user_id_column, 'cosine_distance']].groupby(user_id_column).mean()

    intra_list_diversity = np.mean(results.values.flatten())
    # Calculate support, set it to be the number of users
    support = len(unique_user_ids)

    return intra_list_diversity, support


class IntraListDiversity:
    """Intra-List Diversity@k

    Intra-List Diversity@k measures the intra-list diversity of the recommendations when only k recommendations are
    made to the user. It measures how items from the same user lists are different from each other. This metric has a
    range in :math:`[0, 1]`. The higher this metric is, the more diversified lists of items are recommended to different
    users. Let :math:`U` denote the set of :math:`N` unique users, :math:`u_i`, :math:`u_j \in U` denote the i-th and
    j-th user in the user set, :math:`i, j \in \{0,1,\cdots,N\}`. :math:`R_{u_i}` is the binary indicator vector
    representing provided recommendations for :math:`u_i`. :math:`I` is the set of all unique user pairs,
    :math:`\\forall~i<j, \{u_i, u_j\} \in I`.

    .. math::
            Intra \mbox{-} list~diversity = 1 - \\frac{1}{U}\sum_{i=1}^U average(consine\_similarity(v_p^{u_i}, v_q^{u_i}))
    """

    def __init__(self, item_features: pd.DataFrame, click_column, k: int = None,
                 user_id_column: str = Constants.user_id,
                 item_id_column: str = Constants.item_id, user_sample_size: Union[int, float] = 10000,
                 seed: int = Constants.default_seed, metric: Union[str, Callable] = 'cosine', num_runs: int = 10):
        """Initialize the parameters for Intra-List Diversity metric.

        Parameters
        ----------
        item_features: pd.DataFrame
            features data frame with (item_id, feature_1, feature_2, ..., feature_n) in each row.
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
            num_runs is used to report the approximation of Intra-List Diversity over multiple runs on smaller
            samples of users, default=10, for a speed-up on evaluations. The sampling size is defined by
            user_sample_size. The final result is averaged over the multiple runs.
        
        """

        self.user_id_column = user_id_column
        self.item_id_column = item_id_column
        self.item_features = item_features
        self.click_column = click_column
        self.k = k
        self.user_sample_size = user_sample_size
        self.seed = seed
        self.metric = metric
        self.num_runs = num_runs
        self._validate_arguments()

    def get_score(self, predicted_results: pd.DataFrame, batch_accumulate: bool = False,
                  return_extended_results: bool = False) -> Union[float, dict]:
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
        batch_accumulate: bool
            Should not be True for calculating Intra-List Diversity while it is kept for making the API design
            consistent across different recommender metrics.
        return_extended_results: bool
            Whether the extended results such as the support should also be returned. If specified, the returned results
            will be of type ``dict``. Intra-list diversity currently returns ``Intra-List Diversity`` and
            the ``support``, which is the number of unique users to calculate it.

        Returns
        -------
        metric: Union[float, dict]
            The averaged result(s). The return type is determined by ``return_extended_results`` parameters.
        """

        if batch_accumulate:
            raise ValueError("Batch_accumulate can not be set as True for Intra-List Diversity.")

        results, support = intralist_diversity(predicted_results, self.item_features, self.click_column, self.k,
                                               user_id_column=self.user_id_column,
                                               item_id_column=self.item_id_column,
                                               user_sample_size=self.user_sample_size, seed=self.seed,
                                               metric=self.metric, num_runs=self.num_runs,
                                               )

        if return_extended_results:
            return {'intra-list diversity': results, 'support': support}
        else:
            return results

    def _validate_arguments(self):
        """Validate arguments for Intra-List Diversity"""

        check_true(isinstance(self.num_runs, int), ValueError("num_runs should be an integer."))
        if self.user_sample_size:
            check_true(isinstance(self.user_sample_size, int) or isinstance(self.user_sample_size, float),
                       ValueError("user_sample_size should be an integer or a float number."))
            check_true(self.num_runs >= 1, ValueError("num_runs should be no less than 1."))
            check_true(isinstance(self.num_runs, int), ValueError("num_runs should be an integer."))
        check_true(isinstance(self.click_column, str), ValueError("click_column should be a string."))
        check_true((self.item_id_column in self.item_features.columns),
                   ValueError("item features matrix should have an item id column."))

        if self.k:
            check_true(isinstance(self.k, int), ValueError("k should be an integer."))
        if isinstance(self.user_sample_size, int):
            check_true(self.user_sample_size >= 1, ValueError("user_sample_size should be no less than 1."))
        elif isinstance(self.user_sample_size, float):
            check_true(self.user_sample_size > 0.0, ValueError("user_sample_size should be greater than 0.0."))

    def __str__(self):
        return 'Intra-List Diversity@{}'.format(self.k)

# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Tuple, Optional

import numpy as np
import pandas as pd

from jurity.recommenders.base import _BaseRecommenders
from jurity.utils import Constants, get_sorted_clicks


class CTR(_BaseRecommenders):
    """Click-through rate

    Three supported estimation methods:

    1. Matching
    Calculates the CTR using a direct matching method. That is, CTR is only calculated for instances where the
    actual item the user has seen matches the recommendation.

    2. Inverse Propensity Score (IPS)
    Calculates the IPS, an estimate of CTR with a weighted correction based on how likely an item was to be recommended
    by the historic policy if the user saw the item in the historic data.

    .. math::
        IPS = \frac{1}{n} \sum r_a \times \frac{I(\hat{a} = a)}{p(a|x,h)}

    In this equation:
    * n is the total size of the test data
    * r_a is the observed reward
    * hat{a} is the recommended item
    * I(\hat{a} = a} is a boolean of whether the user-item pair has historic data
    * p(a|x,h) is the probability of the item being recommended for the test context given the historic data

    3. Doubly Robust Estimation (DR)
    Calculates the DR, an estimate of CTR that combines the directly predicted values with a correction based on how
    likely an item was to be recommended by the historic policy if the user saw the item in the historic data.

    ..math::
        DR = \frac{1}{n} \sigma (\hat{r_a} + \frac{(r_a -\hat{r_a}) I(\hat{a} = a}{p(a|x,h)})

    In this equation:
    * n is the total size of the test data
    * r_a is the observed reward
    * \hat{r_a} is the predicted reward
    * hat{a} is the recommended item
    * I(\hat{a} = a} is a boolean of whether the user-item pair has historic data
    * p(a|x,h) is the probability of the item being recommended for the test context given the historic data

    At a high level, doubly robust estimation combines a direct estimate with an IPS-like correction if historic data is
    available. If historic data is not available, the second term is 0 and only the predicted reward is used for the
    user-item pair.

    IPS and DR implementations are based on: Dudík, Miroslav, John Langford, and Lihong Li.
    "Doubly robust policy evaluation and learning." Proceedings of the 28th International Conference on International
    Conference on Machine Learning. 2011. Available as arXiv preprint arXiv:1103.4601 

    """

    def __init__(self, click_column: str, k: Optional[int] = None, user_id_column: str = Constants.user_id,
                 item_id_column: str = Constants.item_id, value_column: Optional[str] = None,
                 estimation: str = 'matching', propensity_column: Optional[str] = Constants.propensity):
        """Initializes the CTR object

        Parameters
        ---------
        click_column: str
            The column to use for scoring the arms.
        k: Optional[int]
            The number of recommendations per user. If not specified, all recommendations will be used.
        user_id_column: str
            The column name for the user ids. Defaults to Constants.user_id.
        item_id_column: str
            The column name for the item ids. Defaults to Constants.item_id.
        value_column: Optional[str]
            The column to calculate the CTR on. If different from ``click_column``, the recommendations will be sorted
            by ``click_column`` and the CTR will be calculated on the matching rows, but on the ``value_column``. If not
            specified, ``click_column`` will be used.
        estimation: str
            The estimation method to use.
            Options: 'matching', 'ips', 'dr'.
            Matching gives a direct estimate of the CTR.
            IPS gives the inverse propensity score.
            DR gives the doubly robust estimate.
        propensity_column: Optional [str]
            The column with historic item probabilities in the actual results, used in IPS and DR.
            Defaults to Constants.propensity.
            If column is not provided, a simple random policy with equal likelihood for every item will be assumed.

        """
        super().__init__(user_id_column=user_id_column, item_id_column=item_id_column)
        self.click_column = click_column
        self.value_column = value_column if value_column else click_column
        self.propensity_column = propensity_column
        self.k = k
        self.estimation = estimation

    def get_score(self, actual_results: pd.DataFrame, predicted_results: pd.DataFrame, batch_accumulate: bool = False,
                  return_extended_results: bool = False) -> Union[float, dict, Tuple[float, float], Tuple[dict, dict]]:
        """Evaluates the current metric on the given data.

        There are 4 scenarios controlled by the ``batch_accumulate`` and ``return_extended_results`` parameters:

        1) Calculating the metric for the whole data:

        This is the default method, which assumes you are operating on the full data and you want to get the metric by
        itself. Returns ``float``.

        .. highlight:: python
        .. code-block:: python

            print(ctr.get_score(actual_responses_batch, recommendations_batch))
            >>> 0.316

        2) Calculating the extended results for the whole data:

        This assumes you are operating on the full data and you want to get the auxiliary information such as the
        support in addition to the metric. The information returned depends on the metric. Returns ``dict``.

        .. highlight:: python
        .. code-block:: python

            print(ctr.get_score(actual_responses_batch, recommendations_batch, return_extended_results=True))
            >>> {'ctr': 0.316, 'support': 122}

        3) Calculating the metric across multiple batches.

        This assumes that you are operating on batched data, and will therefore call this method multiple times for each
        batch. It also assumes that you want to get the metric by itself. Returns ``Tuple[float, float]``.

        .. highlight:: python
        .. code-block:: python

            for actual_responses_batch, recommendations_batch in ..
                ctr_batch, ctr_acc = ctr.get_score(actual_responses_batch, recommendations_batch, accumulate=True)
                print(f'CTR for this batch: {ctr_batch} Overall CTR: {ctr_acc}')
                >>> CTR for this batch: 0.453 Overall CTR: 0.316

        4) Calculating the extended results across multiple matches:

        This assumes you are operating on batched data, and will therefore call this method multiple times for each
        batch. It also assumes you want to get the auxiliary information such as the support in addition to the metric.
        The information returned depends on the metric. Returns ``Tuple[dict, dict]``.

        .. highlight:: python
        .. code-block:: python

            for actual_responses_batch, recommendations_batch in ..
                ctr_batch, ctr_acc = ctr.get_score(actual_responses_batch, recommendations_batch, accumulate=True, return_extended_results=True)
                print(f'CTR for this batch: {ctr_batch} Overall CTR: {ctr_acc}')
                >>> CTR for this batch: {'ctr': 0.453, 'support': 12} Overall CTR: {'ctr': 0.316, 'support': 122}

        Parameters
        ---------
        actual_results: pd.DataFrame
            A pandas DataFrame for the ground truth user item interaction data, captured from historical logs.
            The DataFrame should contain a minimum of two columns, including self._user_id_column, self._item_id_column,
            and anything else the metric may need. Each row contains the interaction of one user with one item, and the
            scores associated with this interaction. There can be multiple interactions per user, and there can be
            multiple users per DataFrame. However, the interactions for a specific user must be contained within a
            single DataFrame.
        predicted_results: pd.DataFrame
            A pandas DataFrame for the recommended user item interaction data, captured from a recommendation algorithm.
            The DataFrame should contain a minimum of two columns, including self._user_id_column, self._item_id_column,
            and anything else the metric may need. Each row contains the interaction of one user with one item, and the
            scores associated with this interaction. There can be multiple interactions per user, and there can be
            multiple users per DataFrame. However, the interactions for a specific user must be contained within a
            single DataFrame.
        batch_accumulate: bool
            If specified, this parameter allows you to pass in minibatches of results and accumulate the metric
            correctly across the batches. This reduces the memory footprint and integrates easily with batched
            training. If specified, the ``get_score`` function will return a tuple of batch results and accumulated
            results.
        return_extended_results: bool
            Whether the extended results such as the support should also be returned. If specified, the returned results
            will be of type ``dict``. CTR currently returns ``ctr`` and the ``support`` used to calculate CTR.

        Returns
        -------
        metric: Union[float, dict, Tuple[float, float], Tuple[dict, dict]]
            The averaged result(s). The return type is determined by the ``batch_accumulate`` and
            ``return_extended_results`` parameters. See the examples above.
        """
        n_items = len(actual_results[self._item_id_column].unique())

        actual_results = actual_results.set_index([self._user_id_column, self._item_id_column])
        predicted_results = predicted_results.set_index([self._user_id_column, self._item_id_column])

        if self.k is not None:
            sorted_clicks = get_sorted_clicks(predicted_results, self._user_id_column, self.click_column, self.k)
        else:
            sorted_clicks = predicted_results

        if self.estimation == 'matching':
            return self._get_matching_ctr(actual_results, sorted_clicks, batch_accumulate, return_extended_results)
        elif self.estimation == 'ips':
            return self._get_ips(actual_results, sorted_clicks, n_items, batch_accumulate, return_extended_results)
        elif self.estimation == 'dr':
            return self._get_doubly_robust_estimate(actual_results, sorted_clicks, n_items, batch_accumulate,
                                                    return_extended_results)
        else:
            raise NotImplementedError('Estimation method not implemented. Must be one of matching, ips, dr.')

    def _get_extended_results(self, results: List[np.ndarray]) -> dict:
        results = np.concatenate(results)
        return {'ctr': np.mean(results), 'support': results.size}

    def _get_matching_ctr(self, actual_results, sorted_clicks, batch_accumulate, return_extended_results):
        matches = self._get_matches(actual_results, sorted_clicks)
        clicks = matches[self.value_column].values

        return self._accumulate_and_return(clicks, batch_accumulate, return_extended_results)

    def _get_ips(self, actual_results, sorted_clicks, n_items, batch_accumulate, return_extended_results):
        matches, actual_results = self._get_match_probabilities(actual_results, sorted_clicks, n_items)

        # calculate reward / propensity for the matches
        matches[Constants.inverse_propensity] = matches[self.value_column] / matches[self.propensity_column]

        # final calculation requires the full n of the data set
        # merge with sorted_clicks and impute with 0 to include the non-matches
        sorted_clicks = sorted_clicks.merge(matches[[Constants.inverse_propensity]],
                                            how='left', left_index=True, right_index=True)
        sorted_clicks[Constants.inverse_propensity].fillna(0, inplace=True)

        ips = sorted_clicks[Constants.inverse_propensity].values

        return self._accumulate_and_return(ips, batch_accumulate, return_extended_results)

    def _get_doubly_robust_estimate(self, actual_results, sorted_clicks, n_items, batch_accumulate,
                                    return_extended_results):
        matches, actual_results = self._get_match_probabilities(actual_results, sorted_clicks, n_items)

        # calculate (actual - predicted) / propensity for the matches
        matches[Constants.ips_correction] = (matches[self.value_column] - matches[self.value_column+'_r']) \
                                           / matches[self.propensity_column]

        # merge with sorted_clicks and impute correction with 0 to include the non-matches for DR calculation
        sorted_clicks = sorted_clicks.merge(matches[[Constants.ips_correction]],
                                            how='left', left_index=True, right_index=True)
        sorted_clicks[Constants.ips_correction].fillna(0, inplace=True)

        # DR = predicted + propensity correction
        sorted_clicks[Constants.estimate] = sorted_clicks[self.value_column] + sorted_clicks[Constants.ips_correction]

        dr = sorted_clicks[Constants.estimate].values

        return self._accumulate_and_return(dr, batch_accumulate, return_extended_results)

    def _get_match_probabilities(self, actual_results, sorted_clicks, n_items):

        if self.propensity_column not in actual_results.columns:

            actual_results[self.propensity_column] = 1/n_items

        matches = self._get_matches(actual_results, sorted_clicks)

        return matches, actual_results

    @staticmethod
    def _get_matches(actual_results, sorted_clicks):
        return actual_results.join(sorted_clicks, how='inner', rsuffix='_r')

    def __str__(self):

        if self.estimation == 'ips':
            estimate_name = 'IPS'
        elif self.estimation == 'dr':
            estimate_name = 'Doubly Robust'
        else:
            estimate_name = 'CTR'
        return estimate_name + '({})@{}'.format(self.value_column, self.k) if self.k is not None \
            else estimate_name + '({})'.format(self.value_column)


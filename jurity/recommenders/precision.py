# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Tuple

import numpy as np
import pandas as pd

from jurity.recommenders.base import _BaseRecommenders
from jurity.utils import Constants, get_sorted_clicks


def precision(actual_results: pd.DataFrame, predicted_results: pd.DataFrame, click_column: str, k: int,
              user_id_column: str = Constants.user_id, item_id_column: str = Constants.item_id):
    # Only consider clicks
    actual_results = actual_results.astype({click_column: bool})
    actual_clicks = actual_results[actual_results[click_column]]

    # Get the users to get_score on, which are the users who have both clicks and predictions
    users = np.intersect1d(actual_clicks[user_id_column].unique(),
                           predicted_results[user_id_column].unique())

    # Sort and get the top predictions
    predicted_results = predicted_results.set_index([user_id_column, item_id_column])
    sorted_clicks = get_sorted_clicks(predicted_results, user_id_column, click_column, k)

    # Merge the predictions and actual clicks together
    merged = sorted_clicks.join(actual_clicks.set_index([user_id_column, item_id_column]), rsuffix='_ac')
    merged = merged.fillna(False)
    merged = merged[merged.index.isin(users, level=0)]  # Only look at users who have both clicks and predictions

    # Get the precision results
    merged_group = merged.groupby(user_id_column)[f'{click_column}_ac']
    results = merged_group.mean().values

    return results


class Precision(_BaseRecommenders):
    """Precision@k

    Precision@k measures the precision of the recommendations when only k recommendations are made to the user. That is,
    it measures the ratio of recommendations among the top k items that are relevant.

    .. math::
        Precision@k = \\frac{1}{\left | A \cap P \\right |}\sum_{i=1}^{\left | A \cap P \\right |} \\frac{\left | A_i \cap P_i[1:k] \\right |}{\left | P_i[1:k] \\right |}

    """

    def __init__(self, click_column, k: int = None, user_id_column: str = Constants.user_id,
                 item_id_column: str = Constants.item_id,  n_items: Union[int, str] = None,
                 n_sampled: Union[int, str] = None):
        super().__init__(user_id_column=user_id_column, item_id_column=item_id_column, n_items=n_items,
                         n_sampled=n_sampled)
        self.click_column = click_column
        self.k = k

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
            will be of type ``dict``. Precision currently returns ``precision`` and the ``support`` used to calculate
            Precision.

        Returns
        -------
        metric: Union[float, dict, Tuple[float, float], Tuple[dict, dict]]
            The averaged result(s). The return type is determined by the ``batch_accumulate`` and
            ``return_extended_results`` parameters. See the examples above.
        """
        results = precision(actual_results, predicted_results, self.click_column, self.k,
                            user_id_column=self._user_id_column, item_id_column=self._item_id_column)
        return self._accumulate_and_return(results, batch_accumulate, return_extended_results)

    def _get_extended_results(self, results: List[np.ndarray]) -> dict:
        results = np.concatenate(results)
        return {'precision': np.mean(results), 'support': results.size}

    def __str__(self):
        return 'Precision@{}'.format(self.k)

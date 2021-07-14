# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Tuple

import numpy as np
import pandas as pd

from jurity.recommenders.base import _BaseRecommenders
from jurity.recommenders.rank_estimation import RankEstimation
from jurity.utils import Constants, get_sorted_clicks

def idcg(num_clicks: int):
    return np.sum(1 / np.log2(np.arange(2, 2 + num_clicks)))


class NDCG(_BaseRecommenders):
    """Normalized Discounted Cumulative Gain

    NDCG measures the ranking of the relevant items with a non-linear, discounted (log2) score per rank. NDCG is
    normalized such that the scores are between 0 and 1.

    .. math::
        NDCG@k = \\frac{1}{\left | A \\right |} \sum_{i=1}^{\left | A \\right |} \\frac {\sum_{r=1}^{\left | P_i \\right |} \\frac{rel(P_{i,r})}{log_2(r+1)}}{\sum_{r=1}^{\left | A_i \\right |} \\frac{1}{log_2(r+1)}}

    """

    def __init__(self, click_column, k: int = None, user_id_column: str = Constants.user_id,
                 item_id_column: str = Constants.item_id, rank_estimation: RankEstimation = None):
        super().__init__(user_id_column=user_id_column, item_id_column=item_id_column)
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
            will be of type ``dict``. NDCG currently returns ``ndcg`` and the ``support`` used to calculate NDCG.

        Returns
        -------
        metric: Union[float, dict, Tuple[float, float], Tuple[dict, dict]]
            The averaged result(s). The return type is determined by the ``batch_accumulate`` and
            ``return_extended_results`` parameters. See the examples above.
        """
        actual_results = actual_results.set_index([self._user_id_column, self._item_id_column])
        predicted_results = predicted_results.set_index([self._user_id_column, self._item_id_column])
        actual_clicks = actual_results[actual_results[self.click_column].astype(bool)]
        idcgs = actual_clicks.groupby(self._user_id_column).count()[self.click_column].apply(idcg)
        sorted_clicks = get_sorted_clicks(predicted_results, self._user_id_column, self.click_column, self.k)
        sorted_clicks['rank'] = sorted_clicks.groupby(self._user_id_column).cumcount() + 1
        reindexed = sorted_clicks.reindex(actual_clicks.index)
        # dcgs = (1. / np.log2(reindexed['rank'] + 1)).groupby(self._user_id_column).sum()
        dcgs = reindexed.groupby(self._user_id_column)['rank'].apply(
            lambda ranks: np.nansum(1 / np.log2(1 + np.array(ranks))))

        ndcgs = (dcgs / idcgs).values
        return self._accumulate_and_return(ndcgs, batch_accumulate, return_extended_results)

    def _get_extended_results(self, results: List[np.ndarray]) -> dict:
        results = np.concatenate(results)
        return {'ndcg': np.mean(results), 'support': results.size}

    def __str__(self):
        return 'NDCG@{}'.format(self.k)

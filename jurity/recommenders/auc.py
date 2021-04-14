# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jurity.recommenders.base import _BaseRecommenders
from jurity.utils import Constants, get_sorted_clicks


class AUC(_BaseRecommenders):
    """Area-Under-the-Curve

    Calculates the AUC using a direct matching method. That is, AUC is calculated for instances where the
    actual item the user has seen matches one of the top-k recommendations.
    """

    def __init__(self, click_column: str, k: Optional[int] = None, user_id_column: str = Constants.user_id,
                 item_id_column: str = Constants.item_id):
        """Initializes the AUC object

        Parameters
        ---------
        click_column: str
            The column name to use for evaluation.
        k: Optional[int]
            The number of recommendations per user. If not specified, all recommendations will be used.
        user_id_column: str
            The column name for the user ids. Defaults to Constants.user_id.
        item_id_column: str
            The column name for the item ids. Defaults to Constants.item_id.
        """
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

            print(auc.get_score(actual_responses, recommendations))
            >>> 0.68

        2) Calculating the extended results for the whole data:

        This assumes you are operating on the full data and you want to get the auxiliary information such as the
        support in addition to the metric. The information returned depends on the metric. Returns ``dict``.

        .. highlight:: python
        .. code-block:: python

            print(auc.get_score(actual_responses, recommendations, return_extended_results=True))
            >>> {'auc': 0.68, 'support': 122}

        3) Calculating the metric across multiple batches.

        This assumes that you are operating on batched data, and will therefore call this method multiple times for each
        batch. It also assumes that you want to get the metric by itself. Returns ``Tuple[float, float]``.

        .. highlight:: python
        .. code-block:: python

            for actual_responses_batch, recommendations_batch in ..
                auc_batch, auc_acc = auc.get_score(actual_responses_batch, recommendations_batch, accumulate=True)
                print(f'AUC for this batch: {auc_batch} Overall AUC: {auc_acc}')
                >>> AUC for this batch: 0.65 Overall AUC: 0.68

        4) Calculating the extended results across multiple matches:

        This assumes you are operating on batched data, and will therefore call this method multiple times for each
        batch. It also assumes you want to get the auxiliary information such as the support in addition to the metric.
        The information returned depends on the metric. Returns ``Tuple[dict, dict]``.

        .. highlight:: python
        .. code-block:: python

            for actual_responses_batch, recommendations_batch in ..
                auc_batch, auc_acc = auc.get_score(actual_responses_batch, recommendations_batch, accumulate=True,
                                                   return_extended_results=True)
                print(f'AUC for this batch: {auc_batch} Overall AUC: {auc_acc}')
                >>> AUC for this batch: {'auc': 0.65, 'support': 12} Overall AUC: {'auc': 0.68, 'support': 122}

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
            will be of type ``dict``. AUC currently returns ``auc`` and the ``support`` used to calculate AUC.

        Returns
        -------
        metric: Union[float, dict, Tuple[float, float], Tuple[dict, dict]]
            The averaged result(s). The return type is determined by the ``batch_accumulate`` and
            ``return_extended_results`` parameters. See the examples above.
        """
        actual_results = actual_results.set_index([self._user_id_column, self._item_id_column])
        predicted_results = predicted_results.set_index([self._user_id_column, self._item_id_column])
        if self.k is not None:
            sorted_clicks = get_sorted_clicks(predicted_results, self._user_id_column, self.click_column, self.k)
        else:
            sorted_clicks = predicted_results

        matches = actual_results.join(sorted_clicks, how='inner', rsuffix='_r')
        clicks = matches[self.click_column].values
        predictions = matches[self.click_column + '_r'].values
        results = np.vstack((clicks, predictions)).T

        return self._accumulate_and_return(results, batch_accumulate, return_extended_results)

    @staticmethod
    def _get_results(results: List[np.ndarray]) -> float:
        results = np.concatenate(results)
        return roc_auc_score(results[:, 0], results[:, 1])

    def _get_extended_results(self, results: List[np.ndarray]) -> dict:
        auc = self._get_results(results)
        return {'auc': auc, 'support': len(np.concatenate(results))}

    def __str__(self):
        if self.k is not None:
            return 'AUC({})@{}'.format(self.click_column, self.k)
        else:
            return 'AUC({})'.format(self.click_column)

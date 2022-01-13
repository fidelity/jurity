# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Tuple

import numpy as np
import pandas as pd

from jurity.recommenders.base import _BaseRecommenders
from jurity.recommenders.interlist_diversity import InterListDiversity


class CombinedMetrics(_BaseRecommenders):
    """Combined Metric Evaluator

    Combines multiple metrics into a single object for ease of use and for sharing computation between different
    metrics.
    """

    def __init__(self, *metrics: _BaseRecommenders):
        super().__init__()
        self.metrics = metrics

    def get_score(self, actual_results: pd.DataFrame, predicted_results: pd.DataFrame, batch_accumulate: bool = False,
                  return_extended_results: bool = False) -> Union[dict, Tuple[dict, dict]]:
        """Evaluates the current metric on the given data.

        There are 4 scenarios controlled by the ``batch_accumulate`` and ``return_extended_results`` parameters:

        1) Calculating the metric for the whole data:

        This is the default method, which assumes you are operating on the full data and you want to get the metric by
        itself. Returns ``dict``.

        ```
        print(metric.get_score(actual_responses_batch, recommendations_batch))
        >>> {'CTR': 0.316, 'NDCG@5': 0.154}
        ```

        2) Calculating the extended results for the whole data:

        This assumes you are operating on the full data and you want to get the auxiliary information such as the
        support in addition to the metric. The information returned depends on the metric. Returns ``dict``.

        ```
        print(metric.get_score(actual_responses_batch, recommendations_batch, return_extended_results=True))
        >>> {'CTR': {'ctr': 0.316, 'support': 122}, 'NDCG@5':{'ndcg': 0.154, 'support': 106}}
        ```

        3) Calculating the metric across multiple batches.

        This assumes that you are operating on batched data, and will therefore call this method multiple times for each
        batch. It also assumes that you want to get the metric by itself. Returns ``Tuple[dict, dict]``.

        ```
        for actual_responses_batch, recommendations_batch in ..
            metrics_batch, metrics_acc = metric.get_score(actual_responses_batch, recommendations_batch, accumulate=True)
            print(f'Metrics for this batch: {metrics_batch} Overall Metrics: {metrics_acc}')
            >>> Metrics for this batch: {'CTR': 0.453, 'NDCG@5': 0.266} Overall Metrics: {'CTR': 0.316, 'NDCG@5': 0.154}
        ```

        4) Calculating the extended results across multiple matches:

        This assumes you are operating on batched data, and will therefore call this method multiple times for each
        batch. It also assumes you want to get the auxiliary information such as the support in addition to the metric.
        The information returned depends on the metric. Returns ``Tuple[dict, dict]``.

        ```
        for actual_responses_batch, recommendations_batch in ..
            metrics_batch, metrics_acc = metric.get_score(actual_responses_batch, recommendations_batch, accumulate=True, return_extended_results=True)
            print(f'Metrics for this batch: {metrics_batch}\nOverall Metrics: {metrics_acc}')
            >>> Metrics for this batch: {'CTR': {'ctr': 0.453, 'support': 23}, 'NDCG@5': {'ndcg': 0.266, 'support': 15}}
            >>> Overall Metrics: {'CTR': {'ctr': 0.316, 'support': 122}, 'NDCG@5': {'ndcg': 0.154, 'support': 106}}
        ```

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
            will be of type ``dict``.

        Returns
        -------
        metric: Union[dict, Tuple[dict, dict]]
            The combined dictionary of results from all metrics. The return type is determined by the
            ``batch_accumulate`` and ``return_extended_results`` parameters. See the examples above.
        """
        batch_vals = dict()
        acc_vals = dict()

        for metric in self.metrics:
            if isinstance(metric, InterListDiversity) and batch_accumulate:
                # Inter-List Diversity requires to know all unique user pairs, thus batch accumulation of the data
                # required for calculating this metric does not fit in this situation.
                raise ValueError("Batch_accumulate can not be set as True when Inter-List Diversity is used in "
                                 "combined metrics.")

        for metric in self.metrics:
            return_val = metric.get_score(actual_results, predicted_results, batch_accumulate,
                                          return_extended_results)
            if return_val is not None:
                if batch_accumulate:
                    batch_val, acc_val = return_val
                    batch_vals[str(metric)] = batch_val
                    acc_vals[str(metric)] = acc_val
                else:
                    acc_vals[str(metric)] = return_val

        if batch_accumulate:
            return batch_vals, acc_vals

        return acc_vals

    def _get_extended_results(self, results: List[np.ndarray]) -> dict:
        return_vals = dict()
        for metric in self.metrics:
            return_vals[str(metric)] = metric._get_extended_results(results)

        return return_vals

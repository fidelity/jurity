# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Union, Tuple, List

import numpy as np
import pandas as pd

from jurity.utils import Constants


class _BaseRecommenders(abc.ABC):
    def __init__(self, user_id_column: str = Constants.user_id,
                 item_id_column: str = Constants.item_id):
        self._results = []
        self._user_id_column = user_id_column
        self._item_id_column = item_id_column

    @abc.abstractmethod
    def get_score(self, actual_results: pd.DataFrame, predicted_results: pd.DataFrame, batch_accumulate: bool = False,
                  return_extended_results: bool = False) -> Union[float, dict, Tuple[float, float], Tuple[dict, dict]]:
        """Evaluates the current metric on the given data.

        There are 4 scenarios controlled by the ``batch_accumulate`` and ``return_extended_results`` parameters:

        1) Calculating the metric for the whole data:

        This is the default method, which assumes you are operating on the full data and you want to get the metric by
        itself. Returns ``float``.

        ```
        print(ctr.get_score(actual_responses_batch, recommendations_batch))
        >>> 0.316
        ```

        2) Calculating the extended results for the whole data:

        This assumes you are operating on the full data and you want to get the auxiliary information such as the
        support in addition to the metric. The information returned depends on the metric. Returns ``dict``.

        ```
        print(ctr.get_score(actual_responses_batch, recommendations_batch, return_extended_results=True))
        >>> {'ctr': 0.316, 'support': 122}
        ```

        3) Calculating the metric across multiple batches.

        This assumes that you are operating on batched data, and will therefore call this method multiple times for each
        batch. It also assumes that you want to get the metric by itself. Returns ``Tuple[float, float]``.

        ```
        for actual_responses_batch, recommendations_batch in ..
            ctr_batch, ctr_acc = ctr.get_score(actual_responses_batch, recommendations_batch, accumulate=True)
            print(f'CTR for this batch: {ctr_batch} Overall CTR: {ctr_acc}')
            >>> CTR for this batch: 0.453 Overall CTR: 0.316
        ```

        4) Calculating the extended results across multiple matches:

        This assumes you are operating on batched data, and will therefore call this method multiple times for each
        batch. It also assumes you want to get the auxiliary information such as the support in addition to the metric.
        The information returned depends on the metric. Returns ``Tuple[dict, dict]``.

        ```
        for actual_responses_batch, recommendations_batch in ..
            ctr_batch, ctr_acc = ctr.get_score(actual_responses_batch, recommendations_batch, accumulate=True, return_extended_results=True)
            print(f'CTR for this batch: {ctr_batch} Overall CTR: {ctr_acc}')
            >>> CTR for this batch: {'ctr': 0.453, 'support': 12} Overall CTR: {'ctr': 0.316, 'support': 122}
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
        metric: Union[float, dict, Tuple[float, float], Tuple[dict, dict]]
            The averaged result(s). The return type is determined by the ``batch_accumulate`` and
            ``return_extended_results`` parameters. See the examples above.
        """
        pass

    @staticmethod
    def _get_results(results: List[np.ndarray]) -> float:
        return np.mean(np.concatenate(results))

    @abc.abstractmethod
    def _get_extended_results(self, results: List[np.ndarray]) -> dict:
        pass

    def _accumulate_and_return(self, results: np.ndarray, accumulate: bool, return_extended_results: bool):
        if results.size == 0:
            cur_result = None
        else:
            if return_extended_results:
                cur_result = self._get_extended_results([results])
            else:
                cur_result = self._get_results([results])
        if accumulate:
            if results.size != 0:
                self._results.append(results)
            if return_extended_results:
                acc_result = self._get_extended_results(self._results)
            else:
                acc_result = self._get_results(self._results)
            return cur_result, acc_result

        return cur_result

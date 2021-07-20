# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Tuple, Optional

import numpy as np
import pandas as pd

from jurity.recommenders.base import _BaseRecommenders
from jurity.utils import Constants, get_sorted_clicks


class IPS(_BaseRecommenders):
    """Inverse Propensity Score

    Calculates the IPS, an estimate of CTR with a weighted correction based on how likely an item was to be recommended
    by the historic policy if the user saw the item in the historic data.

    ..math::
        IPS = \frac{1}{n} \sigma \frac{r_a \times I(\hat{a} = a}{p(a|x,h)}

    In this equation:
    * n is the total size of the test data
    * r_a is the observed reward
    * hat{a} is the recommended item
    * I(\hat{a} = a} is a boolean of whether the user-item pair has historic data
    * p(a|x,h) is the probability of the item being recommended for the test context given the historic data

    """

    def __init__(self, click_column: str, k: Optional[int] = None, user_id_column: str = Constants.user_id,
                 item_id_column: str = Constants.item_id, value_column: Optional[str] = None,
                 propensity_column: Optional[str] = None, n_items: Union[int, str] = None,
                 n_sampled: Union[int, str] = None):
        """Initializes the IPS object

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
            The column to calculate the IPS on. If different from ``click_column``, the recommendations will be sorted
            by ``click_column`` and the IPS will be calculated on the matching rows, but on the ``value_column``. If not
            specified, ``click_column`` will be used.
        propensity_column: Optional [str]
            The column with historic item probabilities in the actual results. Defaults to Constants.propensity_column.
            If column is not provided, a simple random policy with equal likelihood for every item will be assumed.
        n_items: Union[int, str]
            The total number of items that can be recommended.
            Required to use unbiased rank estimation correction for sampled test data.
            If int, the same value will be used for all user ids.
            If string, the column name in actual_results to use for the n_items for each user. For use cases where
            eligibility rules or other filtering mean that not all users can see all items.
            When None, rank estimation correction is not performed.
            Default is None.
        n_sampled: Union[int, str]
            The number of items sampled to create the actual_results test data.
            Required to use unbiased rank estimation correction for sampled test data.
            If int, the same value will be used for all user ids.
            If string, the column name in actual_results to use for the n_sampled for each user. For use cases where
            different numbers of items were sampled for different users.
            When None, rank estimation correction is not performed.
            Default is None.
        """
        pass

    def get_score(self, actual_results: pd.DataFrame, predicted_results: pd.DataFrame, batch_accumulate: bool = False,
                  return_extended_results: bool = False) -> Union[float, dict, Tuple[float, float], Tuple[dict, dict]]:
        """Evaluates the current metric on the given data.

        There are 4 scenarios controlled by the ``batch_accumulate`` and ``return_extended_results`` parameters:

        1) Calculating the metric for the whole data:

        This is the default method, which assumes you are operating on the full data and you want to get the metric by
        itself. Returns ``float``.

        .. highlight:: python
        .. code-block:: python

            print(ips.get_score(actual_responses_batch, recommendations_batch))
            >>> 0.316

        2) Calculating the extended results for the whole data:

        This assumes you are operating on the full data and you want to get the auxiliary information such as the
        support in addition to the metric. The information returned depends on the metric. Returns ``dict``.

        .. highlight:: python
        .. code-block:: python

            print(ips.get_score(actual_responses_batch, recommendations_batch, return_extended_results=True))
            >>> {'ips': 0.316, 'support': 122}

        3) Calculating the metric across multiple batches.

        This assumes that you are operating on batched data, and will therefore call this method multiple times for each
        batch. It also assumes that you want to get the metric by itself. Returns ``Tuple[float, float]``.

        .. highlight:: python
        .. code-block:: python

            for actual_responses_batch, recommendations_batch in ..
                ips_batch, ips_acc = ips.get_score(actual_responses_batch, recommendations_batch, accumulate=True)
                print(f'IPS for this batch: {ips_batch} Overall IPS: {ips_acc}')
                >>> IPS for this batch: 0.453 Overall IPS: 0.316

        4) Calculating the extended results across multiple matches:

        This assumes you are operating on batched data, and will therefore call this method multiple times for each
        batch. It also assumes you want to get the auxiliary information such as the support in addition to the metric.
        The information returned depends on the metric. Returns ``Tuple[dict, dict]``.

        .. highlight:: python
        .. code-block:: python

            for actual_responses_batch, recommendations_batch in ..
                ips_batch, ips_acc = ips.get_score(actual_responses_batch, recommendations_batch, accumulate=True, return_extended_results=True)
                print(f'IPS for this batch: {ips_batch} Overall IPS: {ips_acc}')
                >>> IPS for this batch: {'ips': 0.453, 'support': 12} Overall IPS: {'ips': 0.316, 'support': 122}

        Parameters
        ---------
        actual_results: pd.DataFrame
            A pandas DataFrame for the ground truth user item interaction data, captured from historical logs.
            The DataFrame should contain a minimum of two columns, including self._user_id_column, self._item_id_column,
            and anything else the metric may need. Each row contains the interaction of one user with one item, and the
            scores associated with this interaction. There can be multiple interactions per user, and there can be
            multiple users per DataFrame. However, the interactions for a specific user must be contained within a
            single DataFrame.
            IPS expects a column with the historic probability of each decision in the actual_results dataframe.
            If column is not provided, a simple random policy with equal likelihood for every item will be assumed.
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
            will be of type ``dict``. IPS currently returns ``ips`` and the ``support`` used to calculate IPS.

        Returns
        -------
        metric: Union[float, dict, Tuple[float, float], Tuple[dict, dict]]
            The averaged result(s). The return type is determined by the ``batch_accumulate`` and
            ``return_extended_results`` parameters. See the examples above.
        """
        pass

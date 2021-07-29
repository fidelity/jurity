# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pandas as pd

from jurity.recommenders import BinaryRecoMetrics
from jurity.utils import Constants


class TestBinaryRecommenders(unittest.TestCase):

    def test_ctr(self):
        # Test immediate calculation of CTR
        metric = BinaryRecoMetrics.CTR(click_column='click')
        actual = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                               Constants.item_id: [1, 2, 0, 3],
                               'click': [0, 1, 0, 0]})

        predicted = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                                  Constants.item_id: [1, 2, 2, 3],
                                  'click': [0.8, 0.7, 0.8, 0.7]})

        ctr = metric.get_score(actual, predicted)
        self.assertEqual(1. / 3, ctr)

        # Test accumulated calculation
        metric = BinaryRecoMetrics.CTR(click_column='click')
        _, results = metric.get_score(actual, predicted, batch_accumulate=True, return_extended_results=True)

        self.assertEqual(1. / 3, results['ctr'])
        self.assertEqual(3, results['support'])

    def test_ctr_on_secondary_col(self):
        # Tests calculation of CTR on secondary columns
        actual = pd.DataFrame({Constants.user_id: [1, 2, 2, 3, 4],
                               Constants.item_id: [1, 1, 2, 0, 3],
                               'click': [0, 1, 1, 0, 0],
                               'kpi': [1, 0, 1, 0, 0]})

        predicted = pd.DataFrame({Constants.user_id: [1, 2, 2, 3, 4],
                                  Constants.item_id: [1, 1, 3, 2, 3],
                                  'click': [0.8, 0.7, 0.6, 0.8, 0.7],
                                  'kpi': [1, 0.9, 1, 0, 0]})

        # Make sure the click_column is used for generating the items that are in top k.
        # When using the click column, due to k=1, row 3 should be ignored.
        # Row 4s don't match so they should also be ignored.

        # Select recs by click column, but get_score kpi column
        metric = BinaryRecoMetrics.CTR(click_column='click', k=1, value_column='kpi')
        results = metric.get_score(actual, predicted, return_extended_results=True)

        self.assertEqual(1. / 3, results['ctr'])
        self.assertEqual(3, results['support'])

        # If the value column was being used for selecting the items that are in top k,
        # row 2 of predict would be ignored instead of row 3.
        # Since row 2s don't match, the CTR would only be evaluated on two points.
        metric = BinaryRecoMetrics.CTR(click_column='kpi', k=1, value_column='kpi')
        results = metric.get_score(actual, predicted, return_extended_results=True)

        self.assertEqual(1. / 2, results['ctr'])
        self.assertEqual(2, results['support'])

    def test_ips_ctr(self):
        # Test immediate calculation of CTR
        metric = BinaryRecoMetrics.CTR(click_column='click', estimation='ips')
        actual = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                               Constants.item_id: [1, 2, 0, 3],
                               'click': [0, 1, 0, 0]})

        predicted = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                                  Constants.item_id: [1, 2, 2, 3],
                                  'click': [0.8, 0.7, 0.8, 0.7]})

        ctr = metric.get_score(actual, predicted)
        self.assertEqual(1, ctr)

        # Test accumulated calculation
        metric = BinaryRecoMetrics.CTR(click_column='click', estimation='ips')
        _, results = metric.get_score(actual, predicted, batch_accumulate=True, return_extended_results=True)

        self.assertEqual(1, results['ctr'])
        self.assertEqual(4, results['support'])

    def test_ips_ctr_propensity(self):
        # Test immediate calculation of CTR
        metric = BinaryRecoMetrics.CTR(click_column='click', estimation='ips')
        actual = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                               Constants.item_id: [1, 2, 0, 3],
                               'click': [0, 1, 0, 0],
                               Constants.propensity: [0.1, 0.6, 0.2, 0.1]})

        predicted = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                                  Constants.item_id: [1, 2, 2, 3],
                                  'click': [0.8, 0.7, 0.8, 0.7]})

        ctr = metric.get_score(actual, predicted)
        self.assertEqual(0.4166666666666667, ctr)

        # Test accumulated calculation
        metric = BinaryRecoMetrics.CTR(click_column='click', estimation='ips')
        _, results = metric.get_score(actual, predicted, batch_accumulate=True, return_extended_results=True)

        self.assertEqual(0.4166666666666667, results['ctr'])
        self.assertEqual(4, results['support'])

    def test_dr_ctr(self):
        # Test immediate calculation of CTR
        metric = BinaryRecoMetrics.CTR(click_column='click', estimation='dr')
        actual = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                               Constants.item_id: [1, 2, 0, 3],
                               'click': [0, 1, 0, 0]})

        predicted = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                                  Constants.item_id: [1, 2, 2, 3],
                                  'click': [0.8, 0.7, 0.8, 0.7]})

        ctr = metric.get_score(actual, predicted)
        self.assertEqual(-0.44999999999999996, ctr)

        # Test accumulated calculation
        metric = BinaryRecoMetrics.CTR(click_column='click', estimation='dr')
        _, results = metric.get_score(actual, predicted, batch_accumulate=True, return_extended_results=True)

        self.assertEqual(-0.44999999999999996, results['ctr'])
        self.assertEqual(4, results['support'])

    def test_dr_ctr_propensity(self):
        # Test immediate calculation of CTR
        metric = BinaryRecoMetrics.CTR(click_column='click', estimation='dr')
        actual = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                               Constants.item_id: [1, 2, 0, 3],
                               'click': [0, 1, 0, 0],
                               Constants.propensity: [0.1, 0.6, 0.2, 0.1]})

        predicted = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                                  Constants.item_id: [1, 2, 2, 3],
                                  'click': [0.8, 0.7, 0.8, 0.7]})

        ctr = metric.get_score(actual, predicted)
        self.assertEqual(-2.875, ctr)

        # Test accumulated calculation
        metric = BinaryRecoMetrics.CTR(click_column='click', estimation='dr')
        _, results = metric.get_score(actual, predicted, batch_accumulate=True, return_extended_results=True)

        self.assertEqual(-2.875, results['ctr'])
        self.assertEqual(4, results['support'])

    def test_auc(self):
        # Test immediate calculation of AUC
        metric = BinaryRecoMetrics.AUC(click_column='click')
        actual = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                               Constants.item_id: [1, 2, 0, 3],
                               'click': [0, 1, 0, 0]})

        predicted = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                                  Constants.item_id: [1, 2, 2, 3],
                                  'click': [0.1, 0.9, 0.1, 0.1]})

        auc = metric.get_score(actual, predicted)
        self.assertEqual(1., auc)

        # Test accumulated calculation
        metric = BinaryRecoMetrics.AUC(click_column='click')
        _, results = metric.get_score(actual, predicted, batch_accumulate=True, return_extended_results=True)

        self.assertEqual(1., results['auc'])
        self.assertEqual(3, results['support'])

    def test_auc_extended(self):
        # Test immediate calculation of AUC
        metric = BinaryRecoMetrics.AUC(click_column='click')
        actual = pd.DataFrame({Constants.user_id: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                               Constants.item_id: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                               'click': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]})

        predicted = pd.DataFrame({Constants.user_id: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                                  Constants.item_id: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                                  'click': [0.3, 0.9, 0.4, 0.3, 0.2, 0.9, 0.2, 0.6, 0.9, 0.2, 0.7, 0.8]})

        auc = metric.get_score(actual, predicted)
        self.assertEqual(0.78125, auc)

        # Test accumulated calculation
        metric = BinaryRecoMetrics.AUC(click_column='click')
        _, results = metric.get_score(actual, predicted, batch_accumulate=True, return_extended_results=True)

        self.assertEqual(0.78125, results['auc'])
        self.assertEqual(12, results['support'])

    def test_auc_one_class(self):
        # Test immediate calculation of AUC
        metric = BinaryRecoMetrics.AUC(click_column='click')
        actual = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                               Constants.item_id: [1, 2, 0, 3],
                               'click': [0, 0, 0, 0]})

        predicted = pd.DataFrame({Constants.user_id: [1, 2, 3, 4],
                                  Constants.item_id: [1, 2, 2, 3],
                                  'click': [0.1, 0.9, 0.1, 0.1]})

        auc = metric.get_score(actual, predicted)
        self.assertTrue(np.isnan(auc))

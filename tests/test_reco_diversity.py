# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import pandas as pd
import numpy as np

from jurity.recommenders import DiversityRecoMetrics
from jurity.utils import Constants


class TestDiversityRecommenders(unittest.TestCase):

    actual = pd.DataFrame({Constants.user_id: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                           Constants.item_id: [0, 1, 2, 3,
                                               0, 1, 2, 4,
                                               0, 1, 2, 5],
                           'score': [1, 1, 1, 1,
                                     0, 0, 0, 0,
                                     0, 0, 1, 1]})
    predicted = pd.DataFrame({Constants.user_id: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                              Constants.item_id: [0, 1, 2, 3,
                                                  0, 1, 2, 4,
                                                  0, 1, 2, 5
                                                  ],
                              'score': [0.9, 0.7, 0.6, 0.3,
                                        0.9, 0.7, 0.4, 0.1,
                                        0.9, 0.8, 0.6, 0.6
                                        ]
                              })

    def test_inter_list_diversity_usage(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4)
        results = metric.get_score(self.actual, self.predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(results['inter-list diversity'], 0.25)
        self.assertEqual(results['support'], 3)

    def test_inter_list_diversity_sample_default(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4)
        results = metric.get_score(self.actual, self.predicted, batch_accumulate=False,
                                   return_extended_results=True)

        self.assertEqual(0.25, results['inter-list diversity'])
        self.assertEqual(3, results['support'])

    def test_inter_list_diversity_sample_given(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4, user_sample_size=2)
        results = metric.get_score(self.actual, self.predicted, batch_accumulate=False,
                                   return_extended_results=True)

        self.assertEqual(0.25, results['inter-list diversity'])
        self.assertEqual(2, results['support'])

    def test_inter_list_diversity_sample_one_user(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4, user_sample_size=1)
        results = metric.get_score(self.actual, self.predicted, batch_accumulate=False,
                                   return_extended_results=True)

        self.assertTrue(np.isnan(results['inter-list diversity']))
        self.assertEqual(1, results['support'])

    def test_inter_list_diversity_sample_int_zero(self):
        with self.assertRaises(ValueError):
            # This should fail when user_sample_size is set to be 0.
            metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4, user_sample_size=0)

    def test_inter_list_diversity_sample_float(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4,
                                                         user_sample_size=0.8)
        results = metric.get_score(self.actual, self.predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(0.25, results['inter-list diversity'])
        self.assertEqual(2, results['support'])

    def test_inter_list_diversity_sample_float_zero(self):
        with self.assertRaises(ValueError):
            # This should fail when user_sample_size is set t be 0.0.
            metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4,
                                                             user_sample_size=0.0)

    def test_inter_list_diversity_sample_float_large(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4,
                                                         user_sample_size=1.1)
        results = metric.get_score(self.actual, self.predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(0.25, results['inter-list diversity'])
        self.assertEqual(3, results['support'])

    def test_inter_list_diversity_num_runs_default(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4,
                                                         user_sample_size=0.8)
        results = metric.get_score(self.actual, self.predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(0.25, results['inter-list diversity'])
        self.assertEqual(2, results['support'])

    def test_inter_list_diversity_num_runs_0(self):
        with self.assertRaises(ValueError):
            # This should fail when num_runs=0.
            metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4,
                                                             user_sample_size=0.8, num_runs=0)

    def test_inter_list_diversity_num_runs_float(self):
        with self.assertRaises(ValueError):
            # This should fail when num_runs is a float.
            metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4,
                                                             user_sample_size=0.8, num_runs=0.3)

    def test_inter_list_diversity_batch_accumulate_true(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4)

        with self.assertRaises(ValueError):
            # This should fail when `batch_accumulate=True`.
            results = metric.get_score(self.actual, self.predicted, batch_accumulate=True, return_extended_results=True)






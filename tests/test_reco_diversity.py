# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import pandas as pd
import numpy as np

from jurity.recommenders import DiversityRecoMetrics, RankingRecoMetrics
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

    def test_inter_list_diversity_one(self):
        actual = pd.DataFrame({Constants.user_id: [0, 1],
                               Constants.item_id: [0, 1],
                               'score': [1, 1]})
        predicted = pd.DataFrame({Constants.user_id: [0, 1],
                                  Constants.item_id: [0, 1],
                                  'score': [1, 1]})
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4)
        results = metric.get_score(actual, predicted, batch_accumulate=False,
                                   return_extended_results=True)

        self.assertEqual(results['inter-list diversity'], 1)
        self.assertEqual(2, results['support'])

    def test_inter_list_diversity_zero(self):
        actual = pd.DataFrame({Constants.user_id: [0, 1],
                               Constants.item_id: [0, 1],
                               'score': [1, 1]})
        predicted = pd.DataFrame({Constants.user_id: [0, 1],
                                  Constants.item_id: [1, 1],
                                  'score': [1, 1]})
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4)
        results = metric.get_score(actual, predicted, batch_accumulate=False,
                                   return_extended_results=True)

        self.assertEqual(results['inter-list diversity'], 0)
        self.assertEqual(2, results['support'])

    def test_inter_list_diversity_negative_score(self):
        predicted = pd.DataFrame({Constants.user_id: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                                  Constants.item_id: [0, 1, 2, 3,
                                                      0, 1, 2, 4,
                                                      0, 1, 2, 5
                                                      ],
                                  'score': [-0.9, 0.7, 0.6, 0.3,
                                            0.9, -0.7, 0.4, 0.1,
                                            0.9, 0.8, -0.6, 0.6
                                            ]
                                  })
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4, user_sample_size=3)
        results = metric.get_score(self.actual, predicted, batch_accumulate=False,
                                   return_extended_results=True)

        self.assertEqual(results['inter-list diversity'], 0.25)
        self.assertEqual(3, results['support'])

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

    def test_inter_list_diversity_n_jobs_all(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4, user_sample_size=0.8,
                                                         num_runs=10, n_jobs=-1)
        results = metric.get_score(self.actual, self.predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(results['inter-list diversity'], 0.25)
        self.assertEqual(results['support'], 2)

    def test_inter_list_diversity_working_memory_float(self):
        with self.assertRaises(ValueError):
            # This should fail when working_memory=0.1
            metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4, working_memory=0.1)

    def test_inter_list_diversity_batch_accumulate_true(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4)

        with self.assertRaises(ValueError):
            # This should fail when `batch_accumulate=True`.
            results = metric.get_score(self.actual, self.predicted, batch_accumulate=True, return_extended_results=True)

    def test_inter_list_diversity_metric_euclidean(self):
        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4, user_sample_size=0.8,
                                                         num_runs=10, n_jobs=-1, metric='euclidean')
        results = metric.get_score(self.actual, self.predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(round(results['inter-list diversity'], 4), 1.4142)
        self.assertEqual(results['support'], 2)

        def test_intra_list_diversity_usage(self):
        metric = DiversityRecoMetrics.IntraListDiversity(self.item_features, click_column='score', k=4)
        results = metric.get_score(self.predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(np.round(results['intra-list diversity'], 3), 0.267)
        self.assertEqual(results['support'], 3)

    def test_intra_list_diversity_sample_default(self):
        metric = DiversityRecoMetrics.IntraListDiversity(self.item_features, click_column='score', k=4)
        results = metric.get_score(self.predicted, batch_accumulate=False,
                                   return_extended_results=True)

        self.assertEqual(0.267, np.round(results['intra-list diversity'], 3))
        self.assertEqual(3, results['support'])

    def test_intra_list_diversity_sample_given(self):
        metric = DiversityRecoMetrics.IntraListDiversity(self.item_features, click_column='score', k=4,
                                                         user_sample_size=2)
        results = metric.get_score(self.predicted, batch_accumulate=False,
                                   return_extended_results=True)

        self.assertEqual(0.268, np.round(results['intra-list diversity'], 3))
        self.assertEqual(2, results['support'])

    def test_intra_list_diversity_sample_one_item(self):
        metric = DiversityRecoMetrics.IntraListDiversity(self.item_features, click_column='score', k=1)
        results = metric.get_score(self.predicted, batch_accumulate=False,
                                   return_extended_results=True)

        self.assertTrue(np.isnan(results['intra-list diversity']))
        self.assertEqual(3, results['support'])

    def test_intra_list_diversity_one(self):
        predicted = pd.DataFrame({Constants.user_id: [0, 0],
                                  Constants.item_id: [0, 1],
                                  'score': [1, 1]})
        item_features = pd.DataFrame({Constants.item_id: [0, 1],
                                      'feat1': [0, 1]})

        metric = DiversityRecoMetrics.IntraListDiversity(item_features, click_column='score', k=2)
        results = metric.get_score(predicted, batch_accumulate=False,
                                   return_extended_results=True)

        self.assertEqual(results['intra-list diversity'], 1)
        self.assertEqual(1, results['support'])

    def test_intra_list_diversity_sample_int_zero(self):
        with self.assertRaises(ValueError):
            # This should fail when user_sample_size is set to be 0.
            metric = DiversityRecoMetrics.IntraListDiversity(self.item_features, click_column='score', k=4, user_sample_size=0)

    def test_intra_list_diversity_sample_float(self):
        metric = DiversityRecoMetrics.IntraListDiversity(self.item_features, click_column='score', k=4,
                                                         user_sample_size=0.8)
        results = metric.get_score(self.predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(0.268, np.round(results['intra-list diversity'], 3))
        self.assertEqual(2, results['support'])

    def test_intra_list_diversity_batch_accumulate_true(self):
        metric = DiversityRecoMetrics.IntraListDiversity(self.item_features, click_column='score', k=4)

        with self.assertRaises(ValueError):
            # This should fail when `batch_accumulate=True`.
            results = metric.get_score(self.predicted, batch_accumulate=True, return_extended_results=True)

    def test_intra_list_diversity_metric_euclidean(self):
        metric = DiversityRecoMetrics.IntraListDiversity(self.item_features, click_column='score', k=4,
                                                         metric='euclidean')
        results = metric.get_score(self.predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(round(results['intra-list diversity'], 3), 1.307)
        self.assertEqual(results['support'], 3)

# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import pandas as pd

from jurity.recommenders import DiversityRecoMetrics
from jurity.utils import Constants


class TestDiversityRecommenders(unittest.TestCase):

    def test_inter_list_diversity(self):
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

        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4)
        results = metric.get_score(actual, predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(results['inter-list diversity'], 0.25)
        self.assertEqual(results['support'], 3)

        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4,
                                                         sample_size=0.8)
        results = metric.get_score(actual, predicted, batch_accumulate=False, return_extended_results=True)

        self.assertEqual(0.25, results['inter-list diversity'])
        self.assertEqual(2, results['support'])

        metric = DiversityRecoMetrics.InterListDiversity(click_column='score', k=4,
                                                         chunk_size=1)
        results = metric.get_score(actual, predicted, batch_accumulate=False, return_extended_results=True)
        self.assertEqual(0.25, results['inter-list diversity'])
        self.assertEqual(3, results['support'])




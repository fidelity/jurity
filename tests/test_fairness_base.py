# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings

with warnings.catch_warnings(record=False):
    from jurity.fairness.base import _BaseBinaryFairness


class TestBaseMetrics(unittest.TestCase):
    metric = _BaseBinaryFairness('Test base', 'Test Description', -1, 1, 0)

    def test_base_metric_name(self):
        assert self.metric.name == 'Test base'

    def test_base_metric_description(self):
        assert self.metric.description == 'Test Description'

    def test_base_lower_bound(self):
        assert self.metric.lower_bound == -1

    def test_base_upper_bound(self):
        assert self.metric.upper_bound == 1

    def test_ideal_value(self):
        assert self.metric.ideal_value == 0

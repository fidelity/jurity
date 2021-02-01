# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

from jurity.mitigation.base import _BaseMitigation


class TestBaseMitigation(unittest.TestCase):
    mitigation = _BaseMitigation('Test base', 'Test Description')

    def test_base_mitigation_name(self):
        assert self.mitigation.name == 'Test base'

    def test_base_mitigation_description(self):
        assert self.mitigation.description == 'Test Description'

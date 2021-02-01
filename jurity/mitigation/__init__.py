# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

from jurity.mitigation.base import _BaseMitigation
from .equalized_odds import EqualizedOdds


class BinaryMitigation(NamedTuple):
    """
    Class containing methods for bias mitigation in binary classification tasks.
    """

    EqualizedOdds = EqualizedOdds

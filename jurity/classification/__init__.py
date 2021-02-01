# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0


from typing import NamedTuple

from .accuracy import Accuracy
from .auc import AUC
from .f1 import F1
from .precision import Precision
from .recall import Recall


class BinaryClassificationMetrics(NamedTuple):
    Accuracy = Accuracy
    F1 = F1
    Precision = Precision
    Recall = Recall
    AUC = AUC

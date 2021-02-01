from typing import NamedTuple

from .combined import CombinedMetrics
from .ctr import CTR
from .map import MAP
from .ndcg import NDCG
from .precision import Precision
from .recall import Recall


class BinaryRecoMetrics(NamedTuple):
    CTR = CTR


class RankingRecoMetrics(NamedTuple):
    MAP = MAP
    NDCG = NDCG
    Precision = Precision
    Recall = Recall

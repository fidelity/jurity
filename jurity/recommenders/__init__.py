from typing import NamedTuple

from .auc import AUC
from .combined import CombinedMetrics
from .ctr import CTR
from .map import MAP
from .ndcg import NDCG
from .precision import Precision
from .recall import Recall
from .interlist_diversity import InterListDiversity
from .intralist_diversity import IntraListDiversity


class BinaryRecoMetrics(NamedTuple):
    AUC = AUC
    CTR = CTR


class RankingRecoMetrics(NamedTuple):
    MAP = MAP
    NDCG = NDCG
    Precision = Precision
    Recall = Recall


class DiversityRecoMetrics(NamedTuple):
    InterListDiversity = InterListDiversity
    IntraListDiversity = IntraListDiversity

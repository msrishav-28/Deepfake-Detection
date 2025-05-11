# fusion/__init__.py
from .ensemble.simple_voting import SimpleVoting
from .ensemble.weighted_average import WeightedAverage
from .ensemble.stacking import StackingEnsemble
from .feature_fusion.concat_fusion import ConcatFusion
from .feature_fusion.attention_fusion import AttentionFusion

__all__ = [
    "SimpleVoting",
    "WeightedAverage",
    "StackingEnsemble",
    "ConcatFusion",
    "AttentionFusion"
]
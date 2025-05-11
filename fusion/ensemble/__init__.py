# fusion/ensemble/__init__.py
from .simple_voting import SimpleVoting
from .weighted_average import WeightedAverage
from .stacking import StackingEnsemble

__all__ = ["SimpleVoting", "WeightedAverage", "StackingEnsemble"]
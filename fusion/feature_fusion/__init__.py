# fusion/feature_fusion/__init__.py
from .concat_fusion import ConcatFusion
from .attention_fusion import AttentionFusion

__all__ = ["ConcatFusion", "AttentionFusion"]
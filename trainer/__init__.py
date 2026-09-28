"""Training package for SonoRank and the comparison baselines.

Reader's guide: ``load`` owns subject-disjoint data construction, ``model``
defines the three SonoRank stages and neural baselines, and each ``train_*``
module is a separate command-line entry point. There is no single global
trainer because the stages have different objectives and checkpoint inputs.
"""

__version__ = "0.1.0"

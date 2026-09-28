"""Evaluation package for converting model outputs into paper metrics.

Reader's guide: checkpoint-specific scripts perform inference and save raw
signals, ``event_detection_metrics`` converts those signals into events, and
``evaluate_methods_fair`` applies a common protocol across methods. Plotting
modules consume saved results without rerunning models.
"""

"""
Traversability scoring utilities.
Example: simple rule-based score from segmentation mask and DEM slope.
"""
import numpy as np

def score_traversability(seg_mask: np.ndarray, slope_map: np.ndarray):
    # Simple heuristic: penalize high slope and obstacle pixels
    obstacle_penalty = 10.0
    slope_penalty = 5.0
    score = 100.0
    score -= obstacle_penalty * seg_mask.mean()
    score -= slope_penalty * np.clip(slope_map.mean(), 0, 1)
    return float(max(0.0, score))

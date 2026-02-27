import math
from typing import Iterable


def cosine_similarity(v1: Iterable[float], v2: Iterable[float]) -> float:
    """Calculate cosine similarity between two equal-length vectors.

    Returns 0.0 if either vector has zero magnitude.
    Accepts any iterable of numbers (lists, tuples, generators).
    """
    x = list(v1)
    y = list(v2)

    if len(x) != len(y):
        raise ValueError("Vectors must be the same length")

    dot = sum(a * b for a, b in zip(x, y))
    norm_x = math.sqrt(sum(a * a for a in x))
    norm_y = math.sqrt(sum(b * b for b in y))

    if norm_x == 0 or norm_y == 0:
        return 0.0

    return dot / (norm_x * norm_y)

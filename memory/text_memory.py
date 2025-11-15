import time
import numpy as np
from typing import Dict, List, Any, Tuple


class TextMemory:
    """
    Stores textual observations from OCR:
        text, bbox, center, attention score, embedding, timestamp

    Supports:
      - add(entry)
      - get_recent()
      - find_by_text(query)
      - clear_old(max_age)
    """

    def __init__(self, max_items=200, decay_seconds=20.0):
        self.max_items = max_items
        self.decay_seconds = decay_seconds
        self.items: List[Dict[str, Any]] = []

    def add(self, text: str, bbox, center, score: float, embedding=None):
        timestamp = time.time()

        entry = {
            "text": text,
            "bbox": bbox,
            "center": center,
            "score": score,
            "embedding": embedding,
            "timestamp": timestamp,
        }

        self.items.append(entry)

        # Keep memory bounded
        if len(self.items) > self.max_items:
            self.items.pop(0)

    def clear_old(self):
        """Remove entries older than decay_seconds."""
        now = time.time()
        self.items = [
            it for it in self.items
            if (now - it["timestamp"]) <= self.decay_seconds
        ]

    def get_recent(self) -> List[Dict[str, Any]]:
        """Return still-valid entries."""
        self.clear_old()
        return list(self.items)

    def find_closest_text(self, query: str) -> Dict[str, Any]:
        """
        Naive string similarity for now.
        (Later we integrate text embeddings + patch embeddings.)
        """
        query = query.lower().strip()
        best = None
        best_score = 0.0

        for it in self.get_recent():
            t = it["text"].lower().strip()
            if not t:
                continue
            # simple similarity: shared chars
            sim = self._char_overlap(query, t)
            if sim > best_score:
                best_score = sim
                best = it

        return best

    @staticmethod
    def _char_overlap(a: str, b: str) -> float:
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

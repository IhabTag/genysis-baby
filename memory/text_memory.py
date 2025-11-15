import os
import json
from typing import List, Dict, Any


class TextMemory:
    """
    Very simple text memory:
      - stores a list of screen text entries with bbox, center, score
      - used as a high-level semantic trace of what the agent has seen

    Now with save/load support for persistence.
    """

    def __init__(self, max_items: int = 10000):
        self.max_items = max_items
        self.entries: List[Dict[str, Any]] = []

    # ----------------------------------------------------------
    def add(
        self,
        text: str,
        bbox,
        center,
        score: float,
        embedding=None,
    ):
        """
        Store a single text observation.
        bbox, center should be JSON-serializable (e.g. lists or tuples).
        """
        entry = {
            "text": str(text),
            "bbox": list(bbox) if bbox is not None else None,
            "center": list(center) if center is not None else None,
            "score": float(score),
        }
        # Embedding is optional and currently unused (we set None everywhere)
        if embedding is not None:
            try:
                entry["embedding"] = list(embedding)
            except Exception:
                entry["embedding"] = None

        self.entries.append(entry)

        # Truncate if too large
        if len(self.entries) > self.max_items:
            self.entries = self.entries[-self.max_items :]

    # ----------------------------------------------------------
    def all(self) -> List[Dict[str, Any]]:
        return self.entries

    # ----------------------------------------------------------
    # Persistence helpers
    # ----------------------------------------------------------
    def to_list(self) -> List[Dict[str, Any]]:
        return self.entries

    def from_list(self, data: List[Dict[str, Any]]):
        if data is None:
            self.entries = []
        else:
            self.entries = list(data)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_list(data)

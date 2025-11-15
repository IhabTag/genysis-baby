import random
from typing import Dict, Any


ACTION_TYPES = [
    "MOVE_MOUSE",
    "LEFT_CLICK",
    "RIGHT_CLICK",
    "SCROLL",
    "TYPE_TEXT",
]


def random_action(
    width: int = 1024,
    height: int = 768,
) -> Dict[str, Any]:
    """
    Simple random action generator.

    Returns a dict describing a high-level action on the desktop:
      - MOVE_MOUSE (with random x,y)
      - LEFT_CLICK
      - RIGHT_CLICK
      - SCROLL
      - TYPE_TEXT
    """
    a_type = random.choice(ACTION_TYPES)

    if a_type == "MOVE_MOUSE":
        return {
            "type": "MOVE_MOUSE",
            "x": random.randint(10, width - 10),
            "y": random.randint(10, height - 10),
        }

    if a_type == "LEFT_CLICK":
        return {"type": "LEFT_CLICK"}

    if a_type == "RIGHT_CLICK":
        return {"type": "RIGHT_CLICK"}

    if a_type == "SCROLL":
        return {"type": "SCROLL", "amount": random.randint(-40, 40)}

    if a_type == "TYPE_TEXT":
        txt = random.choice(["test", "hello", "hi", "a", "x"])
        return {"type": "TYPE_TEXT", "text": txt}

    return {"type": "NOOP"}


class RandomAgent:
    """
    Thin wrapper around random_action(), with a simple agent-style API.

    Usage:
      agent = RandomAgent(width=1024, height=768)
      action = agent.select_action(obs)
    """

    def __init__(self, width: int = 1024, height: int = 768):
        self.width = width
        self.height = height

    def select_action(self, obs) -> Dict[str, Any]:
        """
        obs is ignored; action is purely random.
        """
        return random_action(self.width, self.height)

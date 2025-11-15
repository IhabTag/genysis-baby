import time
from typing import Dict, Any

import pyautogui


class ActionExecutor:
    """
    Executes high-level actions on the desktop using pyautogui.

    Supported actions (dicts):
      - {"type": "MOVE_MOUSE", "x": int, "y": int}
      - {"type": "LEFT_CLICK"}
      - {"type": "RIGHT_CLICK"}
      - {"type": "SCROLL", "amount": int}
      - {"type": "TYPE_TEXT", "text": str}
      - {"type": "KEY_PRESS", "key": str}
      - {"type": "NOOP"}
    """

    def __init__(self, width: int = 1024, height: int = 768):
        self.width = width
        self.height = height

        # Disable failsafe (moving to corner won't kill the script)
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.02

    def _safe_call(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            # We don't want the entire environment to crash because of one failed action
            print(f"[ActionExecutor] Warning: action failed: {e}")

    def execute(self, action: Dict[str, Any]) -> None:
        """
        Execute a single action dict.
        """
        if not isinstance(action, dict):
            return

        a_type = action.get("type", "NOOP")

        if a_type == "MOVE_MOUSE":
            x = int(action.get("x", self.width // 2))
            y = int(action.get("y", self.height // 2))
            self._safe_call(pyautogui.moveTo, x, y)
            return

        if a_type == "LEFT_CLICK":
            self._safe_call(pyautogui.click)
            return

        if a_type == "RIGHT_CLICK":
            self._safe_call(pyautogui.click, button="right")
            return

        if a_type == "SCROLL":
            amount = int(action.get("amount", 0))
            self._safe_call(pyautogui.scroll, amount)
            return

        if a_type == "TYPE_TEXT":
            text = action.get("text", "")
            if text:
                self._safe_call(pyautogui.typewrite, text, interval=0.02)
            return

        if a_type == "KEY_PRESS":
            key = action.get("key", None)
            if key:
                self._safe_call(pyautogui.press, key)
            return

        # NOOP or unknown type: do nothing
        return

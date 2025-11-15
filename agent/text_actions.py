import time
from typing import Optional, Dict, Any, List

from memory.text_memory import TextMemory
from agent.screen_interpreter import ScreenInterpreter
from env.actions import ActionExecutor


class TextActionAgent:
    """
    Provides high-level semantic actions using:
      - TextMemory
      - ScreenInterpreter
      - ActionExecutor

    Enables the AGI baby to click UI items by TEXT instead of blind pixel actions.
    """

    def __init__(self, executor: ActionExecutor, interpreter: ScreenInterpreter, memory: TextMemory):
        self.executor = executor
        self.interpreter = interpreter
        self.memory = memory

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _refresh_text_memory(self, frame):
        """Re-run OCR on the frame and update memory."""
        elems = self.interpreter.interpret(frame)
        for e in elems:
            self.memory.add(e["text"], e["bbox"], e["center"], e["score"])

        return elems

    def _click_center_of(self, bbox):
        """Click center of a bounding box."""
        y1, y2, x1, x2 = bbox
        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)

        self.executor.execute({"type": "MOVE_MOUSE", "x": cx, "y": cy})
        time.sleep(0.05)
        self.executor.execute({"type": "LEFT_CLICK"})
        time.sleep(0.05)

    # ============================================================
    # PUBLIC ACTIONS
    # ============================================================

    def find_and_click(self, frame, text: str) -> bool:
        """
        Refresh OCR → find closest match to text → click it.
        Returns True if successful, False otherwise.
        """
        self._refresh_text_memory(frame)
        match = self.memory.find_closest_text(text)

        if match is None:
            print(f"[TextAction] No match for '{text}'.")
            return False

        print(f"[TextAction] Clicking '{match['text']}' at {match['center']}")
        self._click_center_of(match["bbox"])
        return True

    def click_text_near(self, frame, text: str) -> bool:
        """Alias for find_and_click; could be extended for context window."""
        return self.find_and_click(frame, text)

    def type_into(self, frame, target_text: str, content: str) -> bool:
        """
        Clicks the nearest UI element with text matching target_text,
        then types content.
        """
        success = self.find_and_click(frame, target_text)
        if not success:
            return False

        time.sleep(0.1)

        # Type into it
        for ch in content:
            self.executor.execute({"type": "TYPE_TEXT", "text": ch})
            time.sleep(0.03)

        return True

    def get_visible_text(self, frame) -> List[Dict[str, Any]]:
        """
        Returns structured list of all text currently visible.
        """
        elems = self._refresh_text_memory(frame)
        return elems

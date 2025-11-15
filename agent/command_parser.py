import re
from typing import List, Dict, Any, Optional


class CommandParser:
    """
    Step 16:
    A simple rule-based parser that converts natural-language
    instructions into structured action plans.

    Output format:
      [
        {"op": "click", "text": "search"},
        {"op": "type", "text": "editor", "content": "hello world"},
        {"op": "scroll", "amount": -200}
      ]
    """

    def __init__(self):
        pass

    # -----------------------------------------------------------
    # Public
    # -----------------------------------------------------------
    def parse(self, instruction: str) -> List[Dict[str, Any]]:
        instruction = instruction.lower().strip()

        # Split by "and", ";", "then"
        parts = re.split(r"\band\b|;|then", instruction)
        parts = [p.strip() for p in parts if p.strip()]

        plan = []
        for part in parts:
            cmd = self._parse_single(part)
            if cmd:
                plan.append(cmd)

        return plan

    # -----------------------------------------------------------
    # Internal parsing
    # -----------------------------------------------------------
    def _parse_single(self, text: str) -> Optional[Dict[str, Any]]:
        # 1) CLICK
        if "click" in text or "press" in text:
            target = self._extract_target_text(text)
            if target:
                return {"op": "click", "text": target}

        # 2) TYPE INTO
        if "type" in text:
            content = self._extract_quoted(text)
            target = self._extract_target_text(text)
            if not target:
                target = "editor"  # fallback target
            if content:
                return {
                    "op": "type",
                    "text": target,
                    "content": content,
                }

        # 3) SCROLL
        if "scroll" in text:
            if "down" in text:
                return {"op": "scroll", "amount": 200}
            if "up" in text:
                return {"op": "scroll", "amount": -200}
            return {"op": "scroll", "amount": 100}

        return None

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------
    def _extract_quoted(self, text: str) -> Optional[str]:
        """Extract "quoted text" or 'quoted text'."""
        match = re.search(r'"([^"]+)"', text)
        if match:
            return match.group(1)
        match = re.search(r"'([^']+)'", text)
        if match:
            return match.group(1)
        return None

    def _extract_target_text(self, text: str) -> Optional[str]:
        """
        Extract the text after keywords:
          click X
          click on X
          press X
          type into X
          focus X
        """
        tokens = text.split()
        for i, t in enumerate(tokens):
            if t in ["click", "press", "into", "on"]:
                if i + 1 < len(tokens):
                    nxt = tokens[i + 1]
                    if nxt != "the":
                        return nxt.strip().strip('"').strip("'")
                if i + 2 < len(tokens):
                    return tokens[i + 2]
        return None

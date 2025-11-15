import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from env.actions import ActionExecutor


def main():
    executor = ActionExecutor(width=1024, height=768)

    print("Moving mouse to (200, 200)...")
    executor.execute({"type": "MOVE_MOUSE", "x": 200, "y": 200})
    time.sleep(0.5)

    print("Left click...")
    executor.execute({"type": "LEFT_CLICK"})
    time.sleep(0.5)

    print("Typing 'hello'...")
    executor.execute({"type": "TYPE_TEXT", "text": "hello"})
    time.sleep(0.5)

    print("Scrolling down...")
    executor.execute({"type": "SCROLL", "amount": -30})
    time.sleep(0.5)

    print("[test_actions] Finished basic action test.")


if __name__ == "__main__":
    main()

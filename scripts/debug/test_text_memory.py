import os
import sys
import numpy as np
from pprint import pprint

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from env.obs import ScreenCapturer
from agent.screen_interpreter import ScreenInterpreter
from memory.text_memory import TextMemory


def main():
    capt = ScreenCapturer(width=1024, height=768)
    frame = capt.capture()

    interpreter = ScreenInterpreter()
    memory = TextMemory()

    elems = interpreter.interpret(frame)

    print("\n=== Screen Elements ===")
    pprint(elems)

    for elem in elems:
        memory.add(elem["text"], elem["bbox"], elem["center"], elem["score"])

    print("\n=== Text Memory (after insert) ===")
    pprint(memory.get_recent())

    # search test
    q = input("\nSearch text: ").strip()
    match = memory.find_closest_text(q)

    print("\nClosest match:")
    pprint(match)


if __name__ == "__main__":
    main()

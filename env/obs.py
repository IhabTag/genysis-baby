import time
from typing import Optional

import mss
import numpy as np
import cv2


class ScreenCapturer:
    """
    Captures the desktop screen through X/VNC using mss.

    - Grabs from monitor 1 by default (full desktop).
    - Resizes to (width, height).
    - Returns RGB uint8 array of shape (H, W, 3).
    """

    def __init__(self, width: int = 1024, height: int = 768, monitor_index: int = 1):
        self.width = width
        self.height = height

        # mss works with monitors: [0]=all, [1]=primary, ...
        self.sct = mss.mss()
        monitors = self.sct.monitors
        if monitor_index < 0 or monitor_index >= len(monitors):
            monitor_index = 1
        self.monitor = monitors[monitor_index]

        # small delay to make sure X server is ready
        time.sleep(0.2)

    def capture(self) -> np.ndarray:
        """
        Capture screen, convert to RGB, resize, and return as uint8 (H, W, 3).
        """
        img = self.sct.grab(self.monitor)
        arr = np.array(img)              # BGRA
        arr = arr[:, :, :3]              # BGR
        arr = arr[:, :, ::-1]            # BGR -> RGB

        # Resize to desired resolution
        resized = cv2.resize(arr, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return resized.astype(np.uint8)

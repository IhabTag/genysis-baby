import pytesseract
import numpy as np
import cv2
from typing import List, Dict, Any


def run_ocr_on_frame(frame: np.ndarray) -> List[str]:
    """
    Lightweight OCR wrapper.
    
    Inputs:
        frame: RGB uint8 (H,W,3)
    
    Returns:
        List of text strings detected in the frame.
    """
    if frame is None:
        return []

    # Tesseract expects BGR
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    config = "--oem 1 --psm 6"

    text = pytesseract.image_to_string(img_bgr, config=config)

    # Split into lines, filter empty
    lines = [t.strip() for t in text.split("\n") if len(t.strip()) > 0]
    return lines


def run_ocr_with_boxes(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Returns full OCR info including bounding boxes.
    
    Output list of:
        {
            "text": str,
            "bbox": (x1, y1, x2, y2),
            "score": confidence,
            "center": (cx, cy)
        }
    """
    if frame is None:
        return []

    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    config = "--oem 1 --psm 6"
    data = pytesseract.image_to_data(img_bgr, config=config, output_type=pytesseract.Output.DICT)

    results = []

    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        conf = float(data["conf"][i])

        if txt == "" or conf < 30:  # discard low-confidence garbage
            continue

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        bbox = (x, y, x + w, y + h)
        cx, cy = x + w // 2, y + h // 2

        results.append(
            {
                "text": txt,
                "bbox": bbox,
                "center": (cx, cy),
                "score": conf,
            }
        )

    return results

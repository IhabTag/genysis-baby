import numpy as np
from typing import Optional

try:
    import pytesseract
except ImportError:
    pytesseract = None


class OCRReader:
    """
    Thin wrapper around pytesseract for reading text from RGB patches.

    Requires:
      - system package: tesseract-ocr
      - python package: pytesseract
    """

    def __init__(self, lang: str = "eng"):
        if pytesseract is None:
            raise RuntimeError(
                "pytesseract is not installed. "
                "Install with: pip install pytesseract "
                "and ensure 'tesseract-ocr' is installed on the system."
            )
        self.lang = lang

    def read_patch(self, patch: np.ndarray) -> str:
        """
        patch: RGB uint8 image (H,W,3)
        Returns OCR'ed text (stripped).
        """
        assert patch.ndim == 3 and patch.shape[2] == 3, "Expected RGB patch HxWx3"

        # pytesseract works with RGB numpy arrays directly
        text = pytesseract.image_to_string(patch, lang=self.lang)
        text = text.strip()
        return text or ""

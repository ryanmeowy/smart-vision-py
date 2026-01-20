from paddleocr import PaddleOCR
import numpy as np


class OCRService:
    # def __init__(self):
    #     print("🔄 Loading PaddleOCR...")
    #     self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    #     print("✅ PaddleOCR loaded.")

    def extract_text(self, image):
        img_array = np.array(image)
        result = self.ocr.ocr(img_array, cls=True)

        full_text = ""
        lines = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                lines.append(text)
                full_text += text + " "

        return full_text.strip(), lines

ocr_service = OCRService()
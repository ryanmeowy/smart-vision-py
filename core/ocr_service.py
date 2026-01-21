import re

import numpy as np
from paddleocr import PaddleOCR

from utils.image_loader import get_image_smart

min_score = 0.6


def _is_valid_content(text):
    text = text.strip()
    if not text:
        return False
    if len(text) == 1:
        if not re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]', text):
            return False
    if not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', text):
        return False
    return True


class OCRService:
    def __init__(self):
        print("🔄 Loading PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        print("✅ PaddleOCR loaded.")

    def extract_text(self, image_url):
        image = get_image_smart(image_url)
        img_array = np.array(image)
        result = self.ocr.ocr(img_array, cls=True)

        if not result or result[0] is None:
            return "", []

        cleaned_lines = []

        for line_data in result[0]:
            text_info = line_data[1]
            text = text_info[0]
            score = text_info[1]
            if score < min_score:
                continue
            if not _is_valid_content(text):
                continue
            cleaned_lines.append(text)
        full_text = " ".join(cleaned_lines)
        return full_text, cleaned_lines


ocr_service = OCRService()

if __name__ == "__main__":
    service = OCRService()
    # 英文
    url = "https://images.pexels.com/photos/7661135/pexels-photo-7661135.jpeg"
    ocr_result = service.extract_text(url)
    print("OCR Result (temp):", ocr_result)
    print("full_text:", ocr_result[0])
    print("lines:", ocr_result[1])
    # 中文
    url = "https://images.pexels.com/photos/34738471/pexels-photo-34738471.jpeg"
    print("OCR Result (temp):", service.extract_text(url))
    # 复杂背景 + 中文 + 数字
    url = "https://images.pexels.com/photos/34081557/pexels-photo-34081557.jpeg"
    print("OCR Result (temp):", service.extract_text(url))
    # 复杂背景 + 模糊中文
    url = "https://images.pexels.com/photos/32964713/pexels-photo-32964713.jpeg"
    print("OCR Result (temp):", service.extract_text(url))
    # 日文招牌
    url = "https://images.pexels.com/photos/31320539/pexels-photo-31320539.jpeg"
    print("OCR Result (temp):", service.extract_text(url))
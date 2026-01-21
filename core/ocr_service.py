from paddleocr import PaddleOCR
from utils.image_loader import load_image_from_url
import numpy as np


class OCRService:
    def __init__(self):
        print("🔄 Loading PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        print("✅ PaddleOCR loaded.")

    def extract_text(self, image_url):
        image = load_image_from_url(image_url)
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

if __name__ == "__main__":
    service = OCRService()
    # 英文
    url = "https://images.pexels.com/photos/7661135/pexels-photo-7661135.jpeg"
    result = service.extract_text(url)
    print("OCR Result (temp):", result)
    print("full_text:", result[0])
    print("lines:", result[1])
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


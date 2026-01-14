from paddleocr import PaddleOCR
import numpy as np


class OCRService:
    def __init__(self):
        print("🔄 Loading PaddleOCR...")
        # use_angle_cls=True: 支持识别旋转文字
        # lang='ch': 支持中英文
        # show_log=False: 关掉烦人的日志
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        print("✅ PaddleOCR loaded.")

    def extract_text(self, image):
        # PaddleOCR 需要 numpy array 格式
        img_array = np.array(image)

        # 识别
        result = self.ocr.ocr(img_array, cls=True)

        full_text = ""
        lines = []

        # 解析结果 (PaddleOCR 返回结构比较复杂)
        # result 结构: [[[[坐标], [文字, 置信度]], ...]]
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                lines.append(text)
                full_text += text + " "

        return full_text.strip(), lines


# 单例
ocr_service = OCRService()
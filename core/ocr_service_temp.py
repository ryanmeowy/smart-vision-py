from paddleocr import PaddleOCR
import numpy as np


class OCRService:
    def __init__(self):
        print("🔄 Loading PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        print("✅ PaddleOCR loaded.")

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

# ocr_service = OCRService()


from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


if __name__ == "__main__":
    # Load the model
    model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")
    config = load_config("mlx-community/Qwen2-VL-2B-Instruct-4bit")

    # Prepare input
    image = ["https://images.pexels.com/photos/34738471/pexels-photo-34738471.jpeg"]
    prompt = "请精确提取图中的所有文本内容，包括印刷体和清晰的手写体。请忽略水印，并丢弃无意义的文本（如单个标点符号、无上下文的孤立字符）。若图中没有文本、文本无法识别或难以识别，请输出“-1”。若有文本，请直接输出提取到的文本，不要输出任何与图中文本无关的内容。"

    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=1
    )

    # Generate output
    output = generate(model, processor, formatted_prompt, image)
    print(output)
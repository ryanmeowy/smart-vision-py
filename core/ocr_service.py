# ocr_qwen2vl_mlx.py
from mlx_vlm import load, generate
from PIL import Image
import requests
from io import BytesIO
import os


class QwenOcrService:
    # def __init__(self):
    #     print("🔄 Loading Qwen2-VL-2B-Instruct-4bit (MLX)...")
    #     self.model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    #     try:
    #         # 加载 MLX 格式的模型和处理器
    #         self.model, self.processor = load(
    #             self.model_path,
    #             trust_remote_code=True
    #         )
    #         print("✅ Model loaded successfully on Apple Silicon.")
    #     except Exception as e:
    #         print(f"❌ Failed to load model: {e}")
    #         raise

    def _load_image(self, image_url: str) -> Image.Image:
        """统一加载图像：支持本地路径和 HTTP/HTTPS URL"""
        if image_url.startswith(("http://", "https://")):
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            if not os.path.exists(image_url):
                raise FileNotFoundError(f"Local image not found: {image_url}")
            image = Image.open(image_url).convert("RGB")
        return image

    def extract_text(self, image_url: str,
                     prompt: str = "Extract all visible text from the image exactly as it appears.") -> str:
        """
        使用 Qwen2-VL 执行 OCR。
        :param image_url: 图像的本地路径或 HTTP(S) URL
        :param prompt: 提示词（建议明确要求提取文字）
        :return: 模型生成的文本
        """
        try:
            # 1. 加载图像为 PIL Image（必须！）
            image = self._load_image(image_url)

            # 2. 调用 generate —— 注意：不要手动 apply_chat_template！
            # mlx_vlm 内部会自动构建多模态输入
            output = generate(
                model=self.model,
                processor=self.processor,
                image=image,  # ← 必须是 PIL Image
                prompt=prompt,  # ← 纯文本提示
                max_tokens=1024,
                temperature=0.0,  # 低温提高确定性
                repetition_penalty=1.1,
                verbose=False
            )

            return output.strip()

        except Exception as e:
            print(f"❌ OCR inference error: {e}")
            import traceback
            traceback.print_exc()
            return ""


# === 使用示例 ===
if __name__ == "__main__":
    ocr = QwenOcrService()

    # 示例1：本地图片
    text1 = ocr.extract_text("https://images.pexels.com/photos/34738471/pexels-photo-34738471.jpeg", "What text is written in this image?")
    print("OCR Result (local):", text1)

import json
import re
import os

from mlx_vlm import load, generate
from utils.image_loader import load_image_from_url

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def _clean_json_output(text: str):
    """清洗 LLM 返回的 JSON 字符串"""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    if not (text.startswith('{') and text.endswith('}')) and \
            not (text.startswith('[') and text.endswith(']')) and \
            not (text.startswith('"') and text.endswith('"')):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text

class CaptionService:
    def __init__(self):
        self.model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
        print(f"🔄 Loading: {self.model_path} ...")
        self.model, self.processor = load(self.model_path)
        print(f"✅ {self.model_path} loaded")

    def generate_name(self, image_url: str):
        image = load_image_from_url(image_url)
        prompt = """为这张图片起一个3-6字的中文标题，要求美感、简洁、诗意。
        不能有除中文外的其他字符或者标点符号。标题不能超过6个字。
        直接输出标题，不要包含其他字符。
        示例1：
        图片内容：一只橘猫在睡觉
        标题：橘猫午睡
        示例2：
        图片内容：繁华的城市夜景
        标题：城市霓虹"""
        formatted_prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
        )
        output = generate(
            self.model,
            self.processor,
            image=image,
            prompt=formatted_prompt,
            verbose=False,
            max_tokens=10,
            temp=0.5
        )
        return _clean_json_output(output)

    def generate_tags(self, image_url: str):
        image = load_image_from_url(image_url)
        prompt = """分析图片，提取3-5个核心中文标签(物体、场景、风格)。
        严格返回JSON字符串数组，例如：["风景", "雪山", "日落"]。
        不要输出Markdown格式，不要输出任何解释性文字。标签数量不要少于3个"""

        formatted_prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
        )

        output = generate(
            self.model,
            self.processor,
            image=image,
            prompt=formatted_prompt,
            verbose=False,
            max_tokens=200,
            temp=0.7
        )
        return _clean_json_output(output)

    def extract_text(self, image_url: str):
        image = load_image_from_url(image_url)
        prompt = """提取图中的所有文本内容，仅限中文、英文和阿拉伯数字，包括印刷体和清晰的手写体。
        忽略水印，并丢弃无意义的文本，比如如单个标点符号、无上下文的孤立字符。
        若图中没有文本、文本无法识别或难以识别，请输出"-1"。
        若有文本，请直接输出提取到的文本，不要输出任何与图中文本无关的内容。"""

        formatted_prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
        )

        output = generate(
            self.model,
            self.processor,
            image=image,
            prompt=formatted_prompt,
            verbose=False,
            max_tokens=200,
            temp=0.1
        )
        return _clean_json_output(output)

caption_service = CaptionService()


if __name__ == "__main__":
    service = CaptionService()
    url = "https://images.pexels.com/photos/7661135/pexels-photo-7661135.jpeg"

    print("Name:", service.generate_name(url))
    print("Tags:", service.generate_tags(url))

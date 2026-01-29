import json
import os
import re

from mlx_vlm import load, generate

from utils.image_loader import get_image_smart

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _clean_json_output(text: str):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []


def _clean_and_validate_title(text: str) -> str:
    if not text:
        return "未命名图片"

    # --- 移除常见的前缀干扰词 ---
    remove_words = ["标题", "图片", "内容", "名称"]
    for word in remove_words:
        text = text.replace(word, "")

    # --- 正则提取纯中文 ---
    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)

    clean_text = "".join(chinese_chars)

    # --- 长度与空值兜底 ---
    if not clean_text:
        return "未命名图片"

    # 超过6个字截取
    if len(clean_text) > 6:
        clean_text = clean_text[:6]

    if len(clean_text) < 2:
        return "未命名图片"

    return clean_text


def _clean_tags_output(raw_text: str) -> list[str]:
    if not raw_text:
        return []

    try:
        match = re.search(r'\[.*?]', raw_text, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                tags_list = json.loads(json_str)
            except json.JSONDecodeError:
                tags_list = re.findall(r'["\'](.*?)["\']', json_str)
        else:
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            tags_list = re.split(r'[，,、\n]+', clean_text)

        # --- 处理复读机 ---
        seen = set()
        clean_tags = []

        for tag in tags_list:
            if not isinstance(tag, str):
                tag = str(tag)
            tag = tag.strip()

            if not tag or len(tag) > 8:
                continue

            if tag not in seen:
                clean_tags.append(tag)
                seen.add(tag)

        if not clean_tags:
            return ["未分类"]

        return clean_tags[:5]

    except Exception as e:
        print(f"❌ Tags parsing error: {e}, raw: {raw_text}")
        return ["未分类"]


def _clean_graph_triples(text: str):
    try:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()
        matches = re.findall(r'\{[^{}]+}', text)
        triples = []
        seen = set()
        for match in matches:
            try:
                obj = json.loads(match)
                if not all(k in obj for k in ('s', 'p', 'o')):
                    continue
                fingerprint = f"{obj['s']}|{obj['p']}|{obj['o']}"
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    triples.append(obj)
            except:
                continue
        print(f"✅ Graph extracted: {len(triples)} triples")
        return triples
    except Exception as e:
        print(f"❌ Parsing Error: {e}")
        return []


class CaptionService:
    def __init__(self):
        self.model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
        print(f"🔄 Loading: {self.model_path} ...")
        self.model, self.processor = load(self.model_path)
        print(f"✅ {self.model_path} loaded")

    def generate_name(self, image_url: str):
        image = get_image_smart(image_url)
        prompt = """
        为这张图片起一个3-6字的中文标题，要求美感、简洁、诗意。
        不能有除中文外的其他字符或者标点符号。标题不能超过6个字。
        直接输出标题，不要包含其他字符。
        示例1：
        图片内容：一只橘猫在睡觉
        标题：橘猫午睡
        示例2：
        图片内容：繁华的城市夜景
        标题：城市霓虹
        """
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
        return _clean_and_validate_title(output)

    def generate_tags(self, image_url: str):
        image = get_image_smart(image_url)
        prompt = """
        分析图片，提取3-5个核心中文标签(物体、场景、风格)。
        严格返回JSON字符串数组，例如：["风景", "雪山", "日落"]。
        不要输出Markdown格式，不要输出任何解释性文字。标签数量不要少于3个
        """

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
        return _clean_tags_output(output)

    def extract_graph_triples(self, image_url: str):
        image = get_image_smart(image_url)
        prompt = """
                请分析图片，提取图中主要物体之间的 SPO 三元组。
                请以 JSON 数组格式返回，每个元素包含三个字段：
                - "s": Subject (主体，名词)
                - "p": Predicate (关系，如：位于、拿着、穿着、包含，动词/介词)
                - "o": Object (客体，名词)
                
                【示例】：
                输入：一张男人站在山顶看日出的图。
                输出：
                [
                  {"s": "男子", "p": "站在", "o": "山顶"},
                  {"s": "男子", "p": "面向", "o": "太阳"},
                  {"s": "云海", "p": "环绕", "o": "山腰"}
                ]
                
                请输出 JSON 数组，不要Markdown代码块，必须是中文。
        """

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
            max_tokens=256,
            temp=0.3,
            repetition_penalty=1.0,
            do_sample=True,
            top_p=0.9
        )

        return _clean_graph_triples(output)

    def stream_generate(self, image_url: str, prompt: str):
        image = get_image_smart(image_url)

        formatted_prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
        )

        # 使用流式生成模式
        stream_output = generate(
            self.model,
            self.processor,
            image=image,
            prompt=formatted_prompt,
            verbose=False,
            max_tokens=500,
            temp=0.7,
            stream=True  # 启用流式输出
        )

        # 逐个yield生成的token
        for chunk in stream_output:
            yield chunk

    def parse_query_to_graph(self, query: str):
        system_prompt = """
                你是一个搜索意图解析器。请提取用户查询中的【实体关系】，并标准为 JSON 三元组。
                - "s": Subject (主体，名词)
                - "p": Predicate (关系，如：位于、拿着、穿着、包含，动词/介词)
                - "o": Object (客体，名词)

                【示例】：
                输入："找一只在睡觉的橘猫" -> 输出：[{"s":"橘猫", "p":"状态", "o":"睡觉"}, {"s":"橘猫", "p":"颜色", "o":"橘色"}]
                输入："红色的法拉利" -> 输出：[{"s":"法拉利", "p":"颜色", "o":"红色"}]
                输入: "爬雪山的男人" -> 输出: [{"s":"男人", "p":"爬", "o":"雪山"}, {"s":"男人", "p":"动作", "o":"爬"}]
                
                请输出 JSON 数组，不要Markdown代码块，必须是中文。
                """
        full_text_prompt = f"{system_prompt}\n输入：{query}\n输出："
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_text_prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output = generate(
            self.model,
            self.processor,
            prompt=text,
            max_tokens=256,
            temperature=0.1,
            verbose=False
        )
        return _clean_json_output(output)


caption_service = CaptionService()

if __name__ == "__main__":
    service = CaptionService()
    # url = "https://images.pexels.com/photos/5026339/pexels-photo-5026339.jpeg"

    # print("Name:", service.generate_name(url))
    # print("Tags:", service.generate_tags(url))
    # print("Graph Triples:", service.extract_graph_triples(url))
    print(service.parse_query_to_graph("奔跑的男人"))

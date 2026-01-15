import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from threading import Thread

torch.set_num_threads(4)

class CaptionService:
    def __init__(self):
        print("🔄 Loading Qwen2-VL-2B model...")
        self.model_path = "Qwen/Qwen2-VL-2B-Instruct"

        # M1 芯片使用 mps 加速
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # 加载模型 (使用 float16 以节省内存并加速)
        # 注意: M1 对 bf16 支持较好
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )

        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(self.model_path, max_pixels=602112)
        print(f"✅ Qwen2-VL loaded on {self.device}.")

    # def __init__(self):
    #     print("🔄 Loading Qwen2-VL-2B model...")
    #     self.model_path = "Qwen/Qwen2-VL-2B-Instruct"
    #
    #     # ❌ 原来的写法 (会导致 MPS Bug)
    #     # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
    #
    #     # ✅ 修改为：强制使用 CPU (避开 MPS 驱动 Bug)
    #     self.device = "cpu"
    #     print(f"⚠️ Force using device: {self.device} for stability")
    #
    #     # 加载模型
    #     # 注意：CPU 不支持 float16/bfloat16 计算，必须用 float32 (默认)
    #     # 或者使用 "auto" 让它自己选
    #     self.model = Qwen2VLForConditionalGeneration.from_pretrained(
    #         self.model_path,
    #         torch_dtype=torch.float32,  # 让 CPU 自己决定精度 (通常是 float32)
    #         device_map=self.device
    #     )
    #
    #     # 加载处理器
    #     self.processor = AutoProcessor.from_pretrained(self.model_path)
    #     print(f"✅ Qwen2-VL loaded on {self.device}.")

    def stream_generate(self, image_url: str, prompt: str = "请详细描述这张图片"):
        """
        流式生成图片描述
        """
        # 1. 构造消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 2. 预处理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 3. 设置流式输出
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,  # 0.7 比较有创造力，适合写文案
            do_sample=True
        )

        # 4. 在新线程中启动生成 (因为 generate 是阻塞的)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 5. 生成器：不断 yield 新生成的字符
        for new_text in streamer:
            yield new_text

    def generate_text(self, image_url: str, prompt: str):

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def generate_text_list(self, image_url: str, prompt: str, num_sequences: int = 3) -> list[str]:

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 核心修改 1: 设置生成参数
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            num_return_sequences=num_sequences,  # 关键：告诉模型要生成几条
            do_sample=True,  # 关键：必须开启采样，否则生成的几条内容会完全一样
            temperature=0.7  # 可选：控制随机性，越高越发散
        )

        # 核心修改 2: 修复截断逻辑
        input_token_len = inputs.input_ids.shape[1]

        generated_ids_trimmed = [
            out_ids[input_token_len:] for out_ids in generated_ids
        ]

        # 核心修改 3: 批量解码
        # batch_decode 本身就会返回 list[str]
        output_text_list = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text_list


# 单例模式
caption_service = CaptionService()
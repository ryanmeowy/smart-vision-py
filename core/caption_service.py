import torch
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from threading import Thread
import torch.nn.functional as F

torch.set_num_threads(4)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class CaptionService:
    def __init__(self):
        print("🔄 Loading Qwen3-VL-2B model...")
        self.model_path = "Qwen/Qwen3-VL-2B-Instruct"

        # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = "cpu"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            max_pixels=602112,
            trust_remote_code=True)
        print(f"✅ Qwen3-VL loaded on {self.device}.")

    def stream_generate(self, image_url: str, prompt: str = "请详细描述这张图片"):
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

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

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

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def generate_text_list(self, image_url: str, prompt: str) -> list[str]:

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

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            num_return_sequences=1,
            do_sample=True,
            temperature=1
        )

        input_token_len = inputs.input_ids.shape[1]

        generated_ids_trimmed = [
            out_ids[input_token_len:] for out_ids in generated_ids
        ]

        output_text_list = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text_list

    @torch.no_grad()
    def get_embedding(self, text=None, image_url=None):

        messages = []
        content = []
        if image_url:
            content.append({"type": "image", "image": image_url})
        if text:
            content.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": content})

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        embedding = last_hidden_state.mean(dim=1)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.float().cpu().numpy()[0].tolist()


caption_service = CaptionService()

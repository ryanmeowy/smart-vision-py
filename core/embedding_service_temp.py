import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
import os


class EmbeddingService:
    # def __init__(self):
    #     print("🔄 Loading Qwen3-VL-Embedding-2B (Official Native)...")
    #     os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    #     self.model_path = "Qwen/Qwen3-VL-Embedding-2B"
    #
    #     self.device = "mps" if torch.backends.mps.is_available() else "cpu"
    #
    #     print(f"🚀 Embedding Service using device: {self.device}")
    #
    #     self.model = Qwen3VLForConditionalGeneration.from_pretrained(
    #         self.model_path,
    #         torch_dtype=torch.float32,
    #         device_map=self.device,
    #         trust_remote_code=True
    #     )
    #
    #     self.processor = AutoProcessor.from_pretrained(
    #         self.model_path,
    #         min_pixels=256 * 28 * 28,
    #         max_pixels=602112,
    #         trust_remote_code=True
    #     )
    #
    #     self.TARGET_DIM = 1024
    #     print(f"✅ Model loaded. MRL Target Dimension: {self.TARGET_DIM}")

    def _get_embedding(self, messages):
        """
        核心逻辑：Chat Template -> Forward -> Last Token Pooling -> MRL Slicing
        """
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

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]

        seq_len = inputs.input_ids.shape[1]
        embedding = last_hidden_state[0, seq_len - 1, :]

        embedding = embedding[:self.TARGET_DIM]

        embedding = F.normalize(embedding, p=2, dim=0)

        return embedding.float().cpu().numpy().tolist()

    def embed_text(self, text: str):
        instruction = "Retrieve images relevant to the user's query."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction + "\n" + text}
                ]
            }
        ]
        return self._get_embedding(messages)

    def embed_image(self, image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."}  # 占位符，触发视觉编码
                ]
            }
        ]
        return self._get_embedding(messages)


embedding_service = EmbeddingService()
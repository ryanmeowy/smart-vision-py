import torch
from transformers import AltCLIPModel, AltCLIPProcessor
import torch.nn.functional as F
import os


class EmbeddingService:
    # def __init__(self):
    #     print("🔄 Loading AltCLIP model (BAAI)...")
    #     os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    #
    #     self.model_name = "BAAI/AltCLIP"
    #
    #     self.device = "mps" if torch.backends.mps.is_available() else "cpu"
    #     print(f"🚀 Embedding Service using device: {self.device}")
    #
    #     self.model = AltCLIPModel.from_pretrained(self.model_name).to(self.device)
    #     self.processor = AltCLIPProcessor.from_pretrained(self.model_name)
    #     self.model.eval()
    #     print("✅ AltCLIP loaded.")

    @torch.no_grad()
    def embed_text(self, text: str):
        inputs = self.processor(
            text=[text],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        features = self.model.get_text_features(**inputs)

        features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy()[0].tolist()

    @torch.no_grad()
    def embed_image(self, image):
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        features = self.model.get_image_features(**inputs)

        features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy()[0].tolist()


embedding_service = EmbeddingService()

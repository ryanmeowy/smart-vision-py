import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch.nn.functional as F


class EmbeddingService:
    def __init__(self):
        print("🔄 Loading Chinese-CLIP model...")
        model_name = "OFA-Sys/chinese-clip-vit-base-patch16"

        # 自动检测设备 (M1/M2 使用 mps，Nvidia 使用 cuda，否则 cpu)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"🚀 Using device: {self.device}")

        self.model = ChineseCLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = ChineseCLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # 开启评估模式
        print("✅ Chinese-CLIP loaded.")

    @torch.no_grad()  # 不计算梯度，省内存
    def embed_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        # 计算特征
        features = self.model.get_text_features(**inputs)
        # 归一化 (关键！Elasticsearch Cosine Similarity 需要归一化向量)
        features = F.normalize(features, p=2, dim=1)
        # 转回 CPU 并转为 List
        return features.cpu().numpy()[0].tolist()

    @torch.no_grad()
    def embed_image(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()[0].tolist()


# 单例模式
embedding_service = EmbeddingService()
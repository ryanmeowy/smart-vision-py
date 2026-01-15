import torch
from transformers import AltCLIPModel, AltCLIPProcessor
import torch.nn.functional as F
import os


class EmbeddingService:
    def __init__(self):
        print("🔄 Loading AltCLIP model (BAAI)...")
        # 1. 设置国内镜像，防止下载卡死
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 2. 指定模型 ID
        self.model_name = "BAAI/AltCLIP"

        # 3. 强制使用 CPU (M1 上最稳妥、最快的方式，且无兼容性问题)
        self.device = "cpu"
        print(f"🚀 Embedding Service using device: {self.device}")

        # 4. 加载模型
        # AltCLIP 是标准架构，transformers 支持极好
        self.model = AltCLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AltCLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()
        print("✅ AltCLIP loaded.")

    @torch.no_grad()
    def embed_text(self, text: str):
        # 1. 预处理文本
        # padding=True, truncation=True 是标准写法
        inputs = self.processor(
            text=[text],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        # 2. 获取文本特征
        features = self.model.get_text_features(**inputs)

        # 3. 归一化 (Elasticsearch Cosine 必需)
        features = F.normalize(features, p=2, dim=1)

        # 4. 转列表
        return features.cpu().numpy()[0].tolist()

    @torch.no_grad()
    def embed_image(self, image):
        # 1. 预处理图片
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # 2. 获取图片特征
        features = self.model.get_image_features(**inputs)

        # 3. 归一化
        features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy()[0].tolist()


# 单例导出
embedding_service = EmbeddingService()
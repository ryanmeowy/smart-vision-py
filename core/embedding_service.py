import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from utils.image_loader import load_image_from_url


class ChineseClipEmbedder:
    def __init__(self):
        print("🔄 Loading Chinese CLIP (ViT-L/14)...")
        self.model_name = "OFA-Sys/chinese-clip-vit-large-patch14"

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = ChineseCLIPModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        ).to(self.device)
        self.processor = ChineseCLIPProcessor.from_pretrained(self.model_name)

        self.model.eval()
        print("✅ Chinese CLIP loaded.")


    def embed_image(self, image_input):
        image = load_image_from_url(image_input)
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                image_features = self.model.get_image_features(**inputs)
                image_embeddings = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        return image_embeddings.cpu().numpy()

    def embed_text(self, text: str):
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                text_features = self.model.get_text_features(**inputs)
                text_embeddings = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        return text_embeddings.cpu().numpy()

embedding_service = ChineseClipEmbedder()

if __name__ == "__main__":
    embedder = ChineseClipEmbedder()

    img_emb = embedder.embed_image("https://images.pexels.com/photos/1450331/pexels-photo-1450331.jpeg")
    print("Image embedding shape:", img_emb.shape)  # (1, 768)

    txt_emb = embedder.embed_text("橘猫")
    print("Text embedding shape:", txt_emb.shape)  # (1, 768)

    similarity = (txt_emb @ img_emb.T).item()
    print(f"Similarity: {similarity:.4f}")

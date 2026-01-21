import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from utils.image_loader import load_image_from_url


class ChineseClipEmbedder:
    def __init__(self):
        self.model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
        print(f"🔄 Loading {self.model_name}...")

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = ChineseCLIPModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        ).to(self.device)
        self.processor = ChineseCLIPProcessor.from_pretrained(self.model_name)

        self.model.eval()
        print(f"✅ {self.model_name} loaded.")


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
    texts = ["一只橘色的猫", "一只狗", "一辆汽车", "很多美味的食物"]

    print("\n--- 对比测试 ---")
    for text in texts:
        txt_emb = embedder.embed_text(text)
        score = (img_emb @ txt_emb.T).item()
        print(f"橘猫图片 vs 文本\'{text}\': \t{score:.4f}")

    # 同模态
    print("\n--- 同模态(文本) ---")
    txt_emb_a = embedder.embed_text("橘猫")
    txt_emb_b = embedder.embed_text("猫")
    txt_emb_c = embedder.embed_text("一辆汽车")
    txt_emb_d = embedder.embed_text("一只狗")
    txt_emb_e = embedder.embed_text("很多美味的食物")
    print(f"文本'橘猫' vs 文本'猫': \t{(txt_emb_a @ txt_emb_b.T).item():.4f}")
    print(f"文本'橘猫' vs 文本'一辆汽车': \t{(txt_emb_a @ txt_emb_c.T).item():.4f}")
    print(f"文本'橘猫' vs 文本'一只狗': \t{(txt_emb_a @ txt_emb_d.T).item():.4f}")
    print(f"文本'橘猫' vs 文本'很多美味的食物': \t{(txt_emb_a @ txt_emb_e.T).item():.4f}")

    print("\n--- 同模态(图片) ---")
    img_emb_a = embedder.embed_image("https://images.pexels.com/photos/1450331/pexels-photo-1450331.jpeg")
    img_emb_b = embedder.embed_image("https://images.pexels.com/photos/31418533/pexels-photo-31418533.jpeg")
    img_emb_c = embedder.embed_image("https://images.pexels.com/photos/285446/pexels-photo-285446.jpeg")
    img_emb_d = embedder.embed_image("https://images.pexels.com/photos/120049/pexels-photo-120049.jpeg")
    img_emb_e = embedder.embed_image("https://images.pexels.com/photos/1640771/pexels-photo-1640771.jpeg")
    print(f"橘猫图片a vs 橘猫图片b: \t{(img_emb_a @ img_emb_b.T).item():.4f}")
    print(f"橘猫图片a vs 狗图片c: \t{(img_emb_a @ img_emb_c.T).item():.4f}")
    print(f"橘猫图片a vs 汽车图片d: \t{(img_emb_a @ img_emb_d.T).item():.4f}")
    print(f"橘猫图片a vs 食物图片e: \t{(img_emb_a @ img_emb_e.T).item():.4f}")


    # 困难模式文本 (长难句 + 细节描述)
    texts = [
        "一只普通的橘猫",  # 基准
        "一只凶猛的老虎在森林里咆哮",  # 视觉干扰项(颜色像)
        "一只慵懒的橘色猫咪趴在织物上休息",  # 正确的长描述(语义细节)
        "一只正在奔跑跳跃的橘猫",  # 动作不符(语义细节)
    ]

    print(f"\nModel: {embedder.model_name}")
    print(f"--- 困难模式测试 ---")
    for text in texts:
        txt_emb = embedder.embed_text(text)
        score = (img_emb @ txt_emb.T).item()
        print(f"图片 vs '{text[:15]}...': \t{score:.4f}")
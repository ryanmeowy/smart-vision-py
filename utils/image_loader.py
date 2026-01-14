import requests
from PIL import Image
from io import BytesIO

def load_image_from_url(url: str):
    """从 URL 下载图片并转为 PIL Image 对象"""
    try:
        # 设置超时，防止卡死
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to load image from {url}: {e}")
        raise e
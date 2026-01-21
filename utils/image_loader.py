import requests
from PIL import Image
from io import BytesIO

def load_image_from_url(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        max_side = 1024
        if max(image.size) > max_side:
            ratio = max_side / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        print(f"❌ Failed to load image from {url}: {e}")
        raise e
from io import BytesIO
from threading import Lock

import requests
from PIL import Image
from cachetools import TTLCache


def load_image_from_url(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        max_side = 768
        if max(image.size) > max_side:
            ratio = max_side / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        print(f"❌ Failed to load image from {url}: {e}")
        raise e


image_cache = TTLCache(maxsize=100, ttl=60)
cache_lock = Lock()

def get_image_smart(url: str):
    with cache_lock:
        if url in image_cache:
            print(f"⚡️ Cache Hit: {url}")
            return image_cache[url]

    print(f"🌐 Downloading: {url}")
    image = load_image_from_url(url)

    with cache_lock:
        image_cache[url] = image
    return image

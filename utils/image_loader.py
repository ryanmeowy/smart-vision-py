from io import BytesIO
from threading import Lock

import requests
from PIL import Image
from cachetools import TTLCache


def _normalize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    max_side = 768
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_image_from_url(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return _normalize_image(image)
    except Exception as e:
        print(f"❌ Failed to load image from {url}: {e}")
        raise e


def load_image_from_bytes(image_bytes: bytes):
    try:
        if not image_bytes:
            raise ValueError("image bytes is empty")
        image = Image.open(BytesIO(image_bytes))
        return _normalize_image(image)
    except Exception as e:
        print(f"❌ Failed to load image from bytes: {e}")
        raise e


image_cache = TTLCache(maxsize=100, ttl=60 * 3)
cache_lock = Lock()


def get_image_smart(image_input):
    if isinstance(image_input, (bytes, bytearray)):
        return load_image_from_bytes(bytes(image_input))

    if not isinstance(image_input, str):
        raise ValueError("image_input must be URL string or bytes")

    with cache_lock:
        if image_input in image_cache:
            print(f"⚡️ Cache Hit: {image_input}")
            return image_cache[image_input]

    print(f"🌐 Downloading: {image_input}")
    image = load_image_from_url(image_input)

    with cache_lock:
        image_cache[image_input] = image
    return image

from pathlib import Path
from PIL import Image
from collections import Counter

IMAGE_DIR = Path("Manga109-s/images")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

size_counter = Counter()
total_images = 0

for image_path in IMAGE_DIR.rglob("*"):
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in EXTS:
        continue

    with Image.open(image_path) as im:
        width, height = im.size

    size_counter[(width, height)] += 1
    total_images += 1
    if (width, height) == (2380, 1684):
        print(f"IMPOSTER: {image_path.resolve()}")

print(f"Total images: {total_images}")
print(f"Unique image sizes: {len(size_counter)}")
print("-" * 45)

for (width, height), count in size_counter.most_common():
    print(f"{width} x {height}: {count} images")
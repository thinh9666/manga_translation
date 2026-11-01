import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

COLORS = ['#e07a12', '#2674d9', '#d94685', '#21a179']
NAMES = ['text', 'frame', 'face', 'body']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=Path('datasets/detection/manga4'))
    p.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=Path('runs/detection/label_checks'))
    args = p.parse_args()
    images = sorted(x for x in (args.dataset / 'images' / args.split).iterdir() if x.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
    if not images:
        raise RuntimeError('No images found')
    random.Random(args.seed).shuffle(images)
    dst_dir = args.output / args.split
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in images[:args.samples]:
        image = Image.open(path).convert('RGB')
        draw = ImageDraw.Draw(image)
        label_path = args.dataset / 'labels' / args.split / (path.stem + '.txt')
        for line in label_path.read_text().splitlines():
            c, xc, yc, w, h = (float(v) for v in line.split())
            c = int(c)
            iw, ih = image.size
            box = ((xc-w/2)*iw, (yc-h/2)*ih, (xc+w/2)*iw, (yc+h/2)*ih)
            draw.rectangle(box, outline=COLORS[c], width=max(2, iw//500))
            draw.text((box[0], max(0, box[1]-17)), NAMES[c], fill=COLORS[c], stroke_width=1, stroke_fill='white')
        image.save(dst_dir / (path.stem + '.jpg'), quality=90)
    print(f'Saved previews to {dst_dir.resolve()}')

if __name__ == '__main__':
    main()

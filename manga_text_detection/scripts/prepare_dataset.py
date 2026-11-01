import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml

CLASSES = ('text', 'frame', 'face', 'body')
EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def find_image(images_dir, book, index):
    directory = images_dir / book
    for padded in (f'{index:03d}', f'{index:04d}', str(index)):
        for ext in EXTS:
            candidate = directory / (padded + ext)
            if candidate.is_file():
                return candidate
    return None


def split_books(books, val_ratio, test_ratio, seed):
    books = sorted(books)
    if len(books) < 3:
        raise ValueError('Need at least 3 books to make train/val/test splits without book leakage.')
    rng = random.Random(seed)
    rng.shuffle(books)
    n = len(books)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    if n_test + n_val >= n:
        raise ValueError('Split ratios leave no training books.')
    return {'test': books[:n_test], 'val': books[n_test:n_test + n_val], 'train': books[n_test + n_val:]}


def normalize_box(obj, width, height):
    try:
        x1, y1, x2, y2 = (float(obj.attrib[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
    except (KeyError, ValueError):
        return None
    x1 = max(0., min(float(width), x1))
    x2 = max(0., min(float(width), x2))
    y1 = max(0., min(float(height), y1))
    y2 = max(0., min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)


def main():
    p = argparse.ArgumentParser(description='Convert Manga109-s XML to YOLO 4-class detection dataset')
    p.add_argument('--source', type=Path, default = Path('Manga109-s'), help='Folder containing images/ and annotations/')
    p.add_argument('--output', type=Path, default=Path('datasets/detection/manga4'))
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--test-ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--include-empty', action='store_true', help='Include pages without any labeled objects as negatives (may contain unlabeled content!)')
    p.add_argument('--limit-books', type=int, default=None, help='For smoke tests; must be >= 3')
    p.add_argument('--overwrite', action='store_true')
    #store_true tương đương type= bool, default = False
    args = p.parse_args()
    source = args.source.expanduser().resolve() #expanduser(): Thay ~ bằng đường dẫn thư mục home của user.
    output = args.output.expanduser().resolve() #resolve(): Chuyển đường dẫn tương đối thành tuyệt đối, đồng thời xử lý . và ...
    image_root = source / 'images'
    annotation_root = source / 'annotations'
    if not image_root.is_dir() or not annotation_root.is_dir():
        p.error('Source must contain images/ and annotations/ directories')
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        p.error('Invalid split ratios')
    xmls = sorted(annotation_root.glob('*.xml')) #list of Path
    if args.limit_books is not None: #giới hạn số lượng
        xmls = xmls[:args.limit_books]
    if not xmls:
        p.error('No XML files found in annotations/')
    if output.exists(): # check if output path exists
        if not args.overwrite: 
            p.error(f'Output exists: {output}. Use --overwrite to regenerate.')
        shutil.rmtree(output)
    output.mkdir(parents=True) #create path
    assignments = split_books([x.stem for x in xmls], args.val_ratio, args.test_ratio, args.seed)
    #return a dict with {'test': books[:n_test], 'val': books[n_test:n_test + n_val], 'train': books[n_test + n_val:]}
    #x.stem to take file name without extension

    book_to_split = {book: split for split, books in assignments.items() for book in books}
    '''
    book_to_split = {
        "DragonBall": "test",
        "Bleach": "val",
        "Naruto": "train",
        "OnePiece": "train"
    }
    '''

    for split in ('train', 'val', 'test'):
        for sub in ('images', 'labels'):
            (output / sub / split).mkdir(parents=True, exist_ok=True)
            #datasets/manga4/(images,labels)/(train,val,test)/
    stats = {s: {'pages': 0, 'objects': Counter(), 'skipped_unannotated': 0, 
                 'missing_images': 0, 'bad_boxes': 0} for s in assignments}
    #s = train/val/test
    metadata_path = output / 'metadata.jsonl'
    #datasets/manga4/metadata.jsonl

    with metadata_path.open('w', encoding='utf-8') as metadata:
        for xml_path in xmls:
            book = xml_path.stem #book name
            split = book_to_split[book] #test/val or train
            root = ET.parse(xml_path).getroot()
            # ET.parse(xml_path) read XML file and create a ElementTree object represent the tree
            #.getroot() return root element
            for page in root.findall('./pages/page'): #book/pages/page
                index = int(page.attrib['index']) #index of page
                source_img = find_image(image_root, book, index)
                #image_root = Manga109/images
                #source_img = Path of book file, exmp: Manga109-s/images/naruto.jpg
                if source_img is None:
                    stats[split]['missing_images'] += 1
                    print(f"Missing image: book={book}, page={index}")
                    continue
                # Read true image dimensions (XML may be inconsistent with decoded image).
                with Image.open(source_img) as im:
                    width, height = im.size
                labels = []
                objects = []
                for tag_id, name in enumerate(CLASSES):
                    for obj in page.findall(name): #name la text/frame/face/body
                        box = normalize_box(obj, width, height)
                        #x_center, y_center, width, height
                        #
                        if box is None:
                            stats[split]['bad_boxes'] += 1
                            continue
                        labels.append(f'{tag_id} ' + ' '.join(f'{v:.8f}' for v in box))
                        '''labels = [
                            '0 0.40000000 0.50000000 0.20000000 0.10000000',
                            '2 0.30000000 0.20000000 0.10000000 0.10000000'
                        ]'''
                        item = {'type': name, 'id': obj.get('id'), 'bbox_xyxy': [obj.get(k) for k in ('xmin','ymin','xmax','ymax')]}
                        '''
                        item = {
                            'type': 'text',
                            'id': '0000412f',
                            'bbox_xyxy': ['327', '76', '406', '191']
                        }
                        '''
                        if name in ('face','body'):
                            item['character_id'] = obj.get('character')
                        if name == 'text':
                            item['transcription'] = ''.join(obj.itertext()).strip()
                        #'transcription': 'いでよ！ 悪魔っ'

                        objects.append(item)
                        stats[split]['objects'][name] += 1 #stats['train']['objects']['text'] += 1
                if not labels and not args.include_empty:
                    stats[split]['skipped_unannotated'] += 1
                    continue
                filename = f'{book}__{index:04d}{source_img.suffix.lower()}'
                #naruto_001.jpg
                dst_image = output / 'images' / split / filename
                #datasets/manga4/images/(train/val/test)/naruto_001.jpg

                shutil.copy2(source_img, dst_image)
                #coppy image vào dst_image

                (output / 'labels' / split / f'{Path(filename).stem}.txt').write_text('\n'.join(labels) + ('\n' if labels else ''), encoding='utf-8')
                # datasets/manga4/labels/train/naruto_001.txt
                metadata.write(json.dumps({'book': book, 'page_index': index, 'split': split, 'image': str(dst_image.relative_to(output)), 'image_size': [width,height], 'objects': objects}, ensure_ascii=False) + '\n')
                '''
                {
                "book": "Naruto",
                "page_index": 1,
                "split": "train",
                "image": "images/train/Naruto__0001.jpg",
                "image_size": [1654, 1170],
                "objects": []
                }
                objects chứa: 
                item = {
                    'type': 'text',
                    'id': '0000412f',
                    'bbox_xyxy': ['327', '76', '406', '191']
                }
                '''
                stats[split]['pages'] += 1
    cfg = {'path': str(output), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': dict(enumerate(CLASSES))}
    '''
    cfg = {
        'path': 'D:/Projects/datasets/manga4',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'text',
            1: 'frame',
            2: 'face',
            3: 'body'
        }
    }
    '''
    (output / 'data.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
    #datasets/manga4/data.yaml

    (output / 'split_books.json').write_text(json.dumps(assignments, indent=2, ensure_ascii=False), encoding='utf-8')
    #datasets/manga4/split_books.json
    #{'test': books[:n_test], 'val': books[n_test:n_test + n_val], 'train': books[n_test + n_val:]

    serializable_stats = {key: {**value, 'objects': dict(value['objects'])} for key, value in stats.items()}
    (output / 'stats.json').write_text(json.dumps(serializable_stats, indent=2), encoding='utf-8')
    '''
        serializable_stats = {
        'train': {
            'pages': 1000,
            'objects': {
                'text': 5000,
                'frame': 3000
            },
            'skipped_unannotated': 20,
            'missing_images': 0,
            'bad_boxes': 2
        },
        'val': (tương tự)
    }
    '''
    print(json.dumps(serializable_stats, indent=2))
    print(f'Dataset ready: {output / "data.yaml"}')
    if any(v['skipped_unannotated'] for v in stats.values()):
        print('NOTE: Pages with no annotations were excluded to avoid false negatives.')

if __name__ == '__main__':
    main()

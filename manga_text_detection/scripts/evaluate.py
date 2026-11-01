import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('datasets/detection/manga4/data.yaml'))
    p.add_argument('--split', choices=['val','test'], default='test')
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    kwargs = dict(data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz, batch=args.batch, plots=True)
    if args.device is not None:
        kwargs['device'] = args.device
    metrics = YOLO(str(args.weights)).val(**kwargs)
    print(f'mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}')
    names = metrics.names
    for cls, ap in enumerate(metrics.box.maps):
        print(f'{names[cls]}: AP50-95={ap:.4f}')

if __name__ == '__main__':
    main()

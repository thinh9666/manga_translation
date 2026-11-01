import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--source', required=True, help='Image, folder, or video')
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', default=None)
    p.add_argument('--project', default='runs/detection/predict')
    p.add_argument('--name', default='manga4')
    args = p.parse_args()
    kwargs = dict(source=args.source, imgsz=args.imgsz, conf=args.conf, save=True,
                  save_txt=True, save_conf=True, project=args.project, name=args.name)
    if args.device is not None:
        kwargs['device'] = args.device
    YOLO(args.weights).predict(**kwargs)

if __name__ == '__main__':
    main()

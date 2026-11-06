import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('datasets/detection/manga4/data.yaml'))
    p.add_argument('--model', default='yolo26s.pt')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default=None, help='e.g. 0, cpu, 0,1; auto if omitted')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--name', default='yolo26s_1024')
    p.add_argument('--project', default=None) #nơi lưu kết quả train
    p.add_argument('--resume', default=None, help='Path to last.pt to resume interrupted run')
        # Optimizer
    p.add_argument('--optimizer', default='AdamW')
    p.add_argument('--lr0', type=float, default=0.001)
    p.add_argument('--lrf', type=float, default=0.01)

    # Augmentation
    p.add_argument('--mosaic', type=float, default=0.5)
    p.add_argument('--scale', type=float, default=0.2)
    p.add_argument('--close-mosaic', type=int, default=15)
    args = p.parse_args()
    if not args.data.is_file():
        p.error(f'Data YAML not found: {args.data}')
    if args.resume:
        YOLO(args.resume).train(resume=True)
        return
    model = YOLO(args.model)
    config = dict(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        seed=42,

        project=args.project,
        name=args.name,

        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        warmup_epochs=3,

        mosaic=args.mosaic,
        scale=args.scale,
        close_mosaic=args.close_mosaic,

        degrees=0.0,
        flipud=0.0,
        fliplr=0.0,

        save=True,
        plots=True
    )
    #degrees: xoay ảnh tắt
    #flipud : xác xuất lật ảnh dọc = 0
    #fliplr : xs lật ngang = 0
    if args.device is not None:
        config['device'] = args.device
    model.train(**config)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import csv, os, sys
from pathlib import Path

STUDENT_JSON     = Path('/app/STUDENT.json')
INPUT_DIR        = Path('/data/input')
OUTPUT_DIR       = Path('/data/output')
OUTPUT_CSV       = OUTPUT_DIR / 'predictions.csv'
MODEL_PATH       = Path('/app/models/best.onnx')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def cmd_info():
    print(STUDENT_JSON.read_text())


def cmd_predict():
    from detector import CatDetector
    if not MODEL_PATH.exists():
        print(f'[ERROR] Model not found: {MODEL_PATH}', file=sys.stderr)
        sys.exit(1)
    detector = CatDetector(str(MODEL_PATH), imgsz=640, conf=0.25, class_names=('cat',))
    image_paths = []
    for root, _, files in os.walk(INPUT_DIR):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                abs_path = Path(root) / fname
                rel_path = abs_path.relative_to(INPUT_DIR)
                image_paths.append((abs_path, str(rel_path).replace(os.sep, '/')))
    if not image_paths:
        print('[WARN] No images found in /data/input/', file=sys.stderr)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_HEADER = ['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for abs_path, rel_path in image_paths:
            try:
                detections = detector.predict(str(abs_path))
            except Exception as exc:
                print(f'[ERROR] {rel_path}: {exc}', file=sys.stderr)
                detections = []
            if detections:
                for det in detections:
                    writer.writerow({
                        'image_path': rel_path,
                        'xmin':       f"{det['xmin']:.2f}",
                        'ymin':       f"{det['ymin']:.2f}",
                        'xmax':       f"{det['xmax']:.2f}",
                        'ymax':       f"{det['ymax']:.2f}",
                        'confidence': f"{det['confidence']:.4f}",
                        'class':      det['class'],
                    })
            else:
                writer.writerow({'image_path': rel_path, 'xmin': '', 'ymin': '',
                                  'xmax': '', 'ymax': '', 'confidence': '', 'class': ''})
    print(f'[predict] wrote {OUTPUT_CSV}  ({len(image_paths)} images processed)')


def main():
    if len(sys.argv) < 2:
        print('Usage: cli.py <info|predict>', file=sys.stderr); sys.exit(1)
    subcmd = sys.argv[1].strip().lower()
    if subcmd == 'info':      cmd_info()
    elif subcmd == 'predict': cmd_predict()
    else:
        print(f'[ERROR] Unknown subcommand: {subcmd!r}', file=sys.stderr); sys.exit(1)

if __name__ == '__main__':
    main()

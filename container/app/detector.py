import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    """ONNX-based cat detector wrapping a YOLO26 end-to-end export.

    Exported with: model.export(format='onnx', imgsz=640, opset=17, dynamic=False)
    Output shape: (1, 300, 6) -> [x1, y1, x2, y2, score, class]
    """

    def __init__(self, onnx_path, imgsz=640, conf=0.25, class_names=("cat",)):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.imgsz       = imgsz
        self.conf        = conf
        self.class_names = class_names
        self.input_name  = self.session.get_inputs()[0].name
        out_shape = self.session.get_outputs()[0].shape
        print(f"[CatDetector] loaded {onnx_path}")
        print(f"[CatDetector] output shape: {out_shape}  (expected (1, 300, 6))")

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        lb_img, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)
        x = (np.array(lb_img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        raw = self.session.run(None, {self.input_name: x})[0]
        detections = raw[0]
        results = []
        for x1, y1, x2, y2, score, cls in detections:
            if float(score) < self.conf:
                continue
            x1 = (float(x1) - pad_x) / scale
            y1 = (float(y1) - pad_y) / scale
            x2 = (float(x2) - pad_x) / scale
            y2 = (float(y2) - pad_y) / scale
            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))
            cls_idx  = int(cls)
            cls_name = (self.class_names[cls_idx]
                        if cls_idx < len(self.class_names) else str(cls_idx))
            results.append({"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                             "confidence": float(score), "class": cls_name})
        return results

    @staticmethod
    def _letterbox(img, size):
        orig_w, orig_h = img.size
        scale   = min(size / orig_w, size / orig_h)
        new_w   = int(orig_w * scale)
        new_h   = int(orig_h * scale)
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas  = Image.new("RGB", (size, size), (114, 114, 114))
        pad_x   = (size - new_w) // 2
        pad_y   = (size - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        return canvas, scale, (pad_x, pad_y)

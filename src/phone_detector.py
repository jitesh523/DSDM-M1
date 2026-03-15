from ultralytics import YOLO
import cv2
import logging

class PhoneDetector:
    """
    Detects phones in a frame using YOLOv8.
    """
    def __init__(self, model_variant="yolov8n.pt", confidence_threshold=0.3):
        self.logger = logging.getLogger(__name__)
        try:
            self.model = YOLO(model_variant)
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
        self.confidence_threshold = confidence_threshold

    def detect(self, frame, roi=None):
        """
        Detects phones in the given frame or ROI.
        roi: (x, y, w, h)
        """
        if self.model is None:
            return False, []

        target_img = frame
        offset_x, offset_y = 0, 0
        if roi:
            x, y, w, h = roi
            target_img = frame[y:y+h, x:x+w]
            offset_x, offset_y = x, y

        results = self.model(target_img, conf=self.confidence_threshold, verbose=False)
        
        phone_detected = False
        detections = []
        
        for r in results:
            for box in r.boxes:
                # Class 67 in COCO is 'cell phone'
                if int(box.cls) == 67:
                    phone_detected = True
                    b = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                    detections.append({
                        "bbox": [int(b[0] + offset_x), int(b[1] + offset_y), int(b[2] + offset_x), int(b[3] + offset_y)],
                        "conf": float(box.conf)
                    })
                    
        return phone_detected, detections

if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    detector = PhoneDetector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    detected, dets = detector.detect(dummy)
    print(f"Detected: {detected}")

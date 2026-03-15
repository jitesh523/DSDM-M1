import cv2
import time
import logging

class Camera:
    """
    Handles camera capture using OpenCV.
    """
    def __init__(self, source=0, width=640, height=480, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.logger = logging.getLogger(__name__)

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.logger.error(f"Could not open video source: {self.source}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify settings
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.logger.info(f"Camera started: {actual_w}x{actual_h} at {self.fps} FPS")
        return True

    def read(self):
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Failed to read frame")
        return ret, frame

    def stop(self):
        if self.cap:
            self.cap.release()
            self.logger.info("Camera stopped")

if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    cam = Camera()
    if cam.start():
        ret, frame = cam.read()
        if ret:
            print(f"Captured frame shape: {frame.shape}")
        cam.stop()

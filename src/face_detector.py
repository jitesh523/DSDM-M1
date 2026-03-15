import cv2
import mediapipe as mp
import logging

class FaceDetector:
    """
    Handles face detection using MediaPipe.
    """
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        self.logger = logging.getLogger(__name__)

    def detect(self, frame):
        """
        Detects faces in the frame.
        Returns: (face_visible, detections)
        """
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_visible = False
        detections = []
        
        if results.detections:
            face_visible = True
            detections = results.detections
            
        return face_visible, detections

if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    from camera import Camera
    cam = Camera()
    detector = FaceDetector()
    if cam.start():
        ret, frame = cam.read()
        if ret:
            visible, dets = detector.detect(frame)
            print(f"Face visible: {visible}, Detections: {len(dets)}")
        cam.stop()

import cv2
import mediapipe as mp
import numpy as np
import logging

class LandmarkExtractor:
    """
    Extracts facial landmarks using MediaPipe Face Mesh.
    """
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.logger = logging.getLogger(__name__)
        
        # Landmark indices for EAR/MAR/Pose (based on MediaPipe documentation)
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.MOUTH_INDICES = [61, 291, 0, 17, 13, 14] # Coarse mouth
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def extract(self, frame):
        """
        Processes frame and returns 468 landmarks for the first face detected.
        """
        h, w, _ = frame.shape
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array of (x, y) pixels
        landmarks = np.array([
            (int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark
        ])
        
        return landmarks

if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    from camera import Camera
    cam = Camera()
    extractor = LandmarkExtractor()
    if cam.start():
        ret, frame = cam.read()
        if ret:
            landmarks = extractor.extract(frame)
            if landmarks is not None:
                print(f"Extracted {len(landmarks)} landmarks")
        cam.stop()

import cv2
import yaml
import logging
import time
from camera import Camera
from face_detector import FaceDetector
from landmark_extractor import LandmarkExtractor
from eye_detector import EyeDetector
from perclos import PERCLOSManager
from yawn_detector import YawnDetector
from head_pose import HeadPoseEstimator
from gaze_zone import GazeZoneClassifier
from phone_detector import PhoneDetector
from alert_manager import AlertManager
from logger import SystemLogger
from visualizer import Visualizer
from phase3.onnx_inference import ONNXEyeDetector
import argparse

class DMSPipeline:
    def __init__(self, config_path="config/default.yaml", use_onnx=False):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize modules
        self.camera = Camera(fps=self.config['camera']['fps'])
        self.face_detector = FaceDetector()
        self.landmark_extractor = LandmarkExtractor()
        
        self.use_onnx = use_onnx
        if self.use_onnx:
            self.eye_detector = ONNXEyeDetector("models/eye_state_best.onnx")
            self.logger.info("Using Phase 3 ONNX inference for eye state detection.")
        else:
            self.eye_detector = EyeDetector(closed_threshold=self.config['ear']['closed_threshold'])
            self.logger.info("Using baseline EAR logic for eye state detection.")
            
        self.perclos_manager = PERCLOSManager(window_s=self.config['perclos']['window_s'])
        self.yawn_detector = YawnDetector(yawn_threshold=self.config['mar']['yawn_threshold'])
        self.head_pose = HeadPoseEstimator()
        self.gaze_classifier = GazeZoneClassifier()
        self.phone_detector = PhoneDetector()
        self.alert_manager = AlertManager(self.config)
        self.system_logger = SystemLogger()
        self.visualizer = Visualizer(self.config)

    def run(self):
        if not self.camera.start():
            return
            
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                data = {}
                events = {}
                
                # 1. Face & Landmarks
                landmarks = self.landmark_extractor.extract(frame)
                data["landmarks"] = landmarks
                
                if landmarks is not None:
                    # 1.4 Facial landmark extraction
                    # Indices mapping
                    l_eye = landmarks[self.landmark_extractor.LEFT_EYE_INDICES]
                    r_eye = landmarks[self.landmark_extractor.RIGHT_EYE_INDICES]
                    mouth = landmarks[[61, 291, 0, 17, 13, 14]] # map to yawn_detector's expected 6 pts
                    # For pose: nose, chin, l_eye_corner, r_eye_corner, l_mouth, r_mouth
                    pose_pts = landmarks[[1, 152, 33, 263, 61, 291]].astype("float32")
                    
                    # 2. Eye State & EAR (or ONNX probability)
                    if self.use_onnx:
                        # ONNX Model requires frame + landmarks
                        closed_prob, state, blink = self.eye_detector.process(frame, l_eye, r_eye)
                        data["ear"] = 1.0 - closed_prob # use open prob as proxy for visualization
                    else:
                        avg_ear, state, blink = self.eye_detector.process(l_eye, r_eye)
                        data["ear"] = avg_ear
                        
                    data["eye_state"] = state
                    if blink:
                        self.system_logger.log_event("BLINK", duration=blink['duration'])
                    
                    # 3. PERCLOS
                    perclos = self.perclos_manager.update(state == "CLOSED")
                    data["perclos"] = perclos
                    if perclos > self.config['perclos']['closed_pct']:
                        events["drowsy"] = True
                    
                    # 4. Yawn Detection
                    mar, yawn = self.yawn_detector.process(mouth)
                    data["mar"] = mar
                    if yawn:
                        events["yawn"] = True
                        self.system_logger.log_event("YAWN", duration=yawn['duration'])
                        
                    # 5. Head Pose & Gaze
                    pose = self.head_pose.estimate(pose_pts, frame.shape)
                    if pose:
                        data["pose"] = pose
                        zone = self.gaze_classifier.classify(*pose)
                        data["zone"] = zone
                        if zone != "FRONT":
                            events["early_distraction"] = True
                    
                # 6. Phone Detection
                phone_detected, phones = self.phone_detector.detect(frame)
                data["phone_detected"] = phone_detected
                if phone_detected:
                    events["phone_use"] = True
                    
                # 7. Alert Management
                alert_state = self.alert_manager.update(events)
                data["alert_state"] = alert_state
                
                # 8. Visualization
                viz_frame = self.visualizer.draw(frame, data)
                cv2.imshow("DSDM-M1 Pipeline", viz_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSDM-M1 Pipeline")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model for eye state detection")
    args = parser.parse_args()
    
    pipeline = DMSPipeline(use_onnx=args.onnx)
    pipeline.run()

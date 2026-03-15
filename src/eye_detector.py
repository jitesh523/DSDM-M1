import numpy as np
import time
import logging

class EyeDetector:
    """
    Computes Eye Aspect Ratio (EAR) and detects eye state and blinks.
    """
    def __init__(self, closed_threshold=0.20):
        self.closed_threshold = closed_threshold
        self.logger = logging.getLogger(__name__)
        
        # Blink tracking
        self.blink_active = False
        self.blink_start_time = 0
        self.last_state = "OPEN"
        
        # Indices for EAR (relative to the subsets used in landmark_extractor)
        # Assuming we pass eye-specific landmark arrays
        # p1-p4 is horizontal, p2-p6 and p3-p5 are vertical
        # In a 6-point eye set: [0, 1, 2, 3, 4, 5] correspond to [p1, p2, p3, p4, p5, p6]
        self.P1, self.P2, self.P3, self.P4, self.P5, self.P6 = 0, 1, 2, 3, 4, 5

    def compute_ear(self, eye_landmarks):
        """
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        p = eye_landmarks
        # Vertical distances
        v1 = np.linalg.norm(p[self.P2] - p[self.P6])
        v2 = np.linalg.norm(p[self.P3] - p[self.P5])
        # Horizontal distance
        h = np.linalg.norm(p[self.P1] - p[self.P4])
        
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
        return ear

    def process(self, left_eye, right_eye):
        """
        Processes EAR for both eyes and returns state and blink events.
        Returns: (avg_ear, state, blink_event)
        """
        l_ear = self.compute_ear(left_eye)
        r_ear = self.compute_ear(right_eye)
        avg_ear = (l_ear + r_ear) / 2.0
        
        state = "CLOSED" if avg_ear < self.closed_threshold else "OPEN"
        blink_event = None
        
        # Blink detection logic
        if state == "CLOSED" and self.last_state == "OPEN":
            self.blink_active = True
            self.blink_start_time = time.time()
        elif state == "OPEN" and self.last_state == "CLOSED":
            if self.blink_active:
                duration = time.time() - self.blink_start_time
                blink_event = {
                    "start": self.blink_start_time,
                    "end": time.time(),
                    "duration": duration
                }
                self.blink_active = False
                
        self.last_state = state
        return avg_ear, state, blink_event

if __name__ == "__main__":
    # Test with mockup data
    logging.basicConfig(level=logging.INFO)
    detector = EyeDetector(closed_threshold=0.2)
    # Mock open eye landmarks
    eye_open = np.array([[0,0], [1,2], [2,2], [3,0], [2,-2], [1,-2]])
    # Mock closed eye landmarks
    eye_closed = np.array([[0,0], [1,0.1], [2,0.1], [3,0], [2,-0.1], [1,-0.1]])
    
    ear, state, _ = detector.process(eye_open, eye_open)
    print(f"Open State: {state}, EAR: {ear:.4f}")
    
    ear, state, blink = detector.process(eye_closed, eye_closed)
    print(f"Closed State: {state}, EAR: {ear:.4f}")
    
    ear, state, blink = detector.process(eye_open, eye_open)
    if blink:
        print(f"Blink detected! Duration: {blink['duration']:.4f}")

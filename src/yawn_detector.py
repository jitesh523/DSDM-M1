import numpy as np
import time
import logging

class YawnDetector:
    """
    Computes Mouth Aspect Ratio (MAR) and detects yawn events.
    """
    def __init__(self, yawn_threshold=0.6, min_duration=1.0):
        self.yawn_threshold = yawn_threshold
        self.min_duration = min_duration
        self.logger = logging.getLogger(__name__)
        
        # Yawn tracking
        self.yawn_active = False
        self.yawn_start_time = 0
        self.last_state = "CLOSED"

    def compute_mar(self, mouth_landmarks):
        """
        MAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        Simplified MAR using vertical/horizontal ratios.
        In our case, mouth_landmarks are 6 key points: [left, right, top_outer, bottom_outer, top_inner, bottom_inner]
        """
        p = mouth_landmarks
        # Vertical distance (top inner to bottom inner)
        v = np.linalg.norm(p[4] - p[5])
        # Horizontal distance (left to right)
        h = np.linalg.norm(p[0] - p[1])
        
        mar = v / h if h > 0 else 0
        return mar

    def process(self, mouth_landmarks):
        """
        Returns (mar, yawn_event)
        """
        mar = self.compute_mar(mouth_landmarks)
        state = "OPEN" if mar > self.yawn_threshold else "CLOSED"
        yawn_event = None
        
        if state == "OPEN" and self.last_state == "CLOSED":
            self.yawn_active = True
            self.yawn_start_time = time.time()
        elif state == "CLOSED" and self.last_state == "OPEN":
            if self.yawn_active:
                duration = time.time() - self.yawn_start_time
                if duration >= self.min_duration:
                    yawn_event = {
                        "start": self.yawn_start_time,
                        "end": time.time(),
                        "duration": duration
                    }
                self.yawn_active = False
                
        self.last_state = state
        return mar, yawn_event

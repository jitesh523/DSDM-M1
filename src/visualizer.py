import cv2
import numpy as np

class Visualizer:
    """
    Handles real-time debug visualization overlay.
    """
    def __init__(self, config=None):
        self.config = config

    def draw(self, frame, data):
        """
        Draws landmarks, metrics, and alert status on the frame.
        """
        out = frame.copy()
        h, w, _ = out.shape
        
        # Draw Alert Status
        state = data.get("alert_state", "NORMAL")
        color = (0, 255, 0) # Green
        if "EMERGENCY" in state: color = (0, 0, 255) # Red
        elif "URGENT" in state: color = (0, 165, 255) # Orange
        elif "WARNING" in state: color = (0, 255, 255) # Yellow
        
        cv2.putText(out, f"STATUS: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Metrics overlay
        y_offset = 80
        metrics = [
            f"EAR: {data.get('ear', 0):.3f}",
            f"PERCLOS: {data.get('perclos', 0):.2f}",
            f"MAR: {data.get('mar', 0):.3f}",
            f"ZONE: {data.get('zone', 'N/A')}",
            f"PHONE: {'YES' if data.get('phone_detected') else 'NO'}"
        ]
        for m in metrics:
            cv2.putText(out, m, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            
        # Draw landmarks if present
        landmarks = data.get("landmarks")
        if landmarks is not None:
            for pt in landmarks:
                cv2.circle(out, tuple(pt), 1, (0, 255, 0), -1)
                
        # Draw Pose Axes (Simple representation)
        yaw, pitch, roll = data.get("pose", (0, 0, 0))
        cv2.putText(out, f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}", (20, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        
        return out

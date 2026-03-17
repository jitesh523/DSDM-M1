import cv2
import numpy as np
import onnxruntime as ort

class ONNXEyeDetector:
    """
    Eye State (Open/Closed) detection using the ONNX model.
    Replaces the rule-based EAR EyeDetector for Phase 3.
    """
    def __init__(self, model_path="models/eye_state_best.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Blink tracking
        self.blink_active = False
        self.blink_start_time = 0
        self.last_state = "OPEN"
        
    def _crop_eye(self, frame, eye_landmarks, margin=0.2):
        """
        Crop the eye from the frame based on landmarks and return the preprocessed tensor.
        """
        if eye_landmarks is None or len(eye_landmarks) == 0:
            return None
            
        x_min, y_min = np.min(eye_landmarks, axis=0)
        x_max, y_max = np.max(eye_landmarks, axis=0)
        
        w = x_max - x_min
        h = y_max - y_min
        
        # Add margin
        x_min = max(0, int(x_min - w * margin))
        x_max = min(frame.shape[1], int(x_max + w * margin))
        y_min = max(0, int(y_min - h * margin))
        y_max = min(frame.shape[0], int(y_max + h * margin))
        
        if y_max <= y_min or x_max <= x_min:
            return None
            
        crop = frame[y_min:y_max, x_min:x_max]
        if len(crop.shape) == 3: # Convert to grayscale if necessary
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
        # Resize to 32x32 as expected by the model
        crop_resized = cv2.resize(crop, (32, 32))
        
        # Normalize and add batch/channel dims -> (1, 1, 32, 32)
        tensor = crop_resized.astype("float32") / 255.0
        tensor = np.expand_dims(np.expand_dims(tensor, axis=0), axis=0)
        return tensor

    def process(self, frame, left_eye, right_eye):
        """
        Run inference on both eye crops and average the logits to determine state.
        Returns: (confidence, state, blink_event)
        """
        import time
        
        l_crop = self._crop_eye(frame, left_eye)
        r_crop = self._crop_eye(frame, right_eye)
        
        if l_crop is None or r_crop is None:
            return 0.0, "OPEN", None
            
        # Run inference
        l_out = self.session.run(None, {self.input_name: l_crop})[0]
        r_out = self.session.run(None, {self.input_name: r_crop})[0]
        
        # Model output is [Closed_Logit, Open_Logit]
        # Average the logits
        avg_logits = (l_out[0] + r_out[0]) / 2.0
        
        # Softmax for probabilities
        exp_logits = np.exp(avg_logits - np.max(avg_logits))
        probs = exp_logits / exp_logits.sum()
        
        closed_prob = probs[0]
        state = "CLOSED" if closed_prob > 0.5 else "OPEN"
        
        blink_event = None
        # Blink detection logic reusing Phase 1 transition logic
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
        return closed_prob, state, blink_event

import os
if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "../../models/eye_state_best.onnx")
    detector = ONNXEyeDetector(model_path)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_l_eye = np.array([[100, 100], [120, 100], [120, 110], [100, 110]])
    dummy_r_eye = np.array([[200, 100], [220, 100], [220, 110], [200, 110]])
    conf, state, blink = detector.process(dummy_frame, dummy_l_eye, dummy_r_eye)
    print(f"Test Run -> State: {state}, Closed Prob: {conf:.4f}")

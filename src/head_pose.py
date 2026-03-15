import cv2
import numpy as np
import logging

class HeadPoseEstimator:
    """
    Estimates head pose (yaw, pitch, roll) using solvePnP.
    """
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        self.logger = logging.getLogger(__name__)
        
        # 3D generic face model points (approximate)
        # Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

    def estimate(self, image_points, frame_shape):
        """
        Estimates yaw, pitch, roll from 2D image points.
        image_points: 6 landmarks mapping to model_points
        """
        h, w, _ = frame_shape
        
        if self.camera_matrix is None:
            # Approximate camera matrix if not provided
            focal_length = w
            center = (w / 2, h / 2)
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
            
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rotation_vector)
        
        # Get Euler angles
        # projection_matrix = np.hstack((rmat, translation_vector))
        # _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        
        # Manual extraction to be sure of order (Yaw, Pitch, Roll)
        # These depend on camera and coordinate system orientation
        # This is a common mapping for frontal face
        pitch = np.arcsin(-rmat[2, 0])
        yaw = np.arctan2(rmat[2, 1], rmat[2, 2])
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
        
        # Convert to degrees
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

if __name__ == "__main__":
    # Test with mockup front-facing points
    estimator = HeadPoseEstimator()
    # Dummy image points (assume 640x480)
    pts = np.array([
        (320, 240), # Nose
        (320, 400), # Chin
        (250, 200), # L eye
        (390, 200), # R eye
        (280, 350), # L mouth
        (360, 350)  # R mouth
    ], dtype=np.float32)
    
    pose = estimator.estimate(pts, (480, 640, 3))
    print(f"Yaw: {pose[0]:.2f}, Pitch: {pose[1]:.2f}, Roll: {pose[2]:.2f}")

import logging

class GazeZoneClassifier:
    """
    Classifies gaze zone based on head pose angles.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def classify(self, yaw, pitch, roll):
        """
        Coarse zone classification based on thresholds.
        """
        # Blueprint thresholds (approximate)
        if abs(yaw) < 15 and abs(pitch) < 15:
            return "FRONT"
        elif yaw < -30:
            return "LEFT_MIRROR"
        elif yaw > 30:
            return "RIGHT_MIRROR"
        elif yaw > 15 and pitch > 15:
            return "INFOTAINMENT"
        elif pitch > 25:
            return "LAP_PHONE"
        else:
            return "OTHER_SIDE"

if __name__ == "__main__":
    classifier = GazeZoneClassifier()
    print(f"Yaw 0, Pitch 0: {classifier.classify(0, 0, 0)}")
    print(f"Yaw 45, Pitch 0: {classifier.classify(45, 0, 0)}")
    print(f"Yaw 0, Pitch 30: {classifier.classify(0, 30, 0)}")

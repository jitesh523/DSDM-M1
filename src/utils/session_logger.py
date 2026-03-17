import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path

class SessionLogger:
    """
    Persists pipeline events and metrics to structured files for post-session analysis.
    """
    def __init__(self, base_log_dir="logs/sessions"):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(base_log_dir) / self.session_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.log_dir / "events.csv"
        self.metadata_path = self.log_dir / "metadata.json"
        
        # Initialize CSV with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type", "value", "confidence", "duration"])
            
        # Write initial metadata
        self._write_metadata({"start_time": self.session_id, "status": "running"})
        
    def _write_metadata(self, data):
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=4)

    def log_event(self, event_type, value="", confidence=1.0, duration=0.0):
        """
        Log a specific event to the session CSV.
        """
        timestamp = time.time()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_type, value, confidence, duration])

    def finalize(self, stats=None):
        """
        Finalize session metadata with summary stats.
        """
        metadata = {
            "session_id": self.session_id,
            "end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status": "completed",
            "summary_stats": stats or {}
        }
        self._write_metadata(metadata)
        print(f"Session logs saved to {self.log_dir}")

if __name__ == "__main__":
    # Test logger
    logger = SessionLogger("../../logs/test_sessions")
    logger.log_event("EYE_STATE", "CLOSED", confidence=0.85)
    logger.log_event("BLINK", duration=0.25)
    logger.log_event("ALERT", "DROWSINESS_WARNING")
    logger.finalize({"total_blinks": 1})

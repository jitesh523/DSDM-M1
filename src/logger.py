import json
import csv
import os
import time
import logging

class SystemLogger:
    """
    Logs system events and metrics to JSON and CSV.
    """
    def __init__(self, log_dir="data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.json_file = os.path.join(log_dir, f"session_{self.session_id}.json")
        self.csv_file = os.path.join(log_dir, f"session_{self.session_id}.csv")
        
        # Initialize CSV
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type", "severity", "duration", "meta"])
            
        self.logger = logging.getLogger(__name__)

    def log_event(self, event_type, severity="INFO", duration=0, meta=None):
        timestamp = time.time()
        entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "severity": severity,
            "duration": duration,
            "meta": meta or {}
        }
        
        # Log to JSON (append-style list would be better, but this is simple for prototype)
        with open(self.json_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
            
        # Log to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_type, severity, duration, json.dumps(meta)])
            
        self.logger.info(f"Logged Event: {event_type} ({severity})")

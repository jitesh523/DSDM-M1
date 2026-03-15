import time
import logging

class AlertManager:
    """
    Manages multi-stage alerts using a state machine.
    ALERTS: NORMAL -> SOFT -> MEDIUM -> URGENT -> EMERGENCY
    """
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.state = "NORMAL"
        self.last_alert_time = 0
        self.cooldown_s = config.get("alert", {}).get("cooldown_s", 15.0) if config else 15.0
        
        # Risk scores (0-1)
        self.distraction_risk = 0.0
        self.drowsiness_risk = 0.0
        
    def update(self, events):
        """
        Processes detected events and updates alert state.
        events: dict of active/recent detection flags
        """
        now = time.time()
        new_state = "NORMAL"
        
        # High urgency events take precedence
        if events.get("sleep") or events.get("unresponsive"):
            new_state = "EMERGENCY"
        elif events.get("microsleep"):
            new_state = "URGENT"
        elif events.get("long_distraction") or events.get("vats_alert"):
            new_state = "MEDIUM_WARNING"
        elif events.get("early_distraction") or events.get("phone_use"):
            new_state = "SOFT_WARNING"
            
        # Cooldown & Escalation logic
        if self.is_more_severe(new_state, self.state):
            # Escalate immediately if more severe
            self.state = new_state
            self.last_alert_time = now
        elif now - self.last_alert_time > self.cooldown_s:
            # Drop back to lower level after cooldown
            self.state = new_state
            
        return self.state

    def is_more_severe(self, s1, s2):
        levels = ["NORMAL", "SOFT_WARNING", "MEDIUM_WARNING", "URGENT", "EMERGENCY"]
        return levels.index(s1) > levels.index(s2)

if __name__ == "__main__":
    manager = AlertManager()
    print(f"Initial: {manager.state}")
    print(f"Update (microsleep): {manager.update({'microsleep': True})}")
    print(f"Update (normal): {manager.update({})}")
    time.sleep(1) # Still in cooldown
    print(f"Update (normal, wait 1s): {manager.update({})}")

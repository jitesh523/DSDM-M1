import time
import collections

class PERCLOSManager:
    """
    Computes PERCLOS (Percentage of eye closure) over a rolling window.
    """
    def __init__(self, window_s=60, closed_pct_threshold=0.80):
        self.window_s = window_s
        self.closed_pct_threshold = closed_pct_threshold
        # Stores (timestamp, is_closed)
        self.history = collections.deque()

    def update(self, is_closed):
        """
        Adds current eye state and returns current PERCLOS.
        """
        now = time.time()
        self.history.append((now, float(is_closed)))
        
        # Remove old samples
        while self.history and (now - self.history[0][0] > self.window_s):
            self.history.popleft()
            
        if not self.history:
            return 0.0
            
        # Compute PERCLOS
        closed_count = sum(sample[1] for sample in self.history)
        perclos = closed_count / len(self.history)
        return perclos

if __name__ == "__main__":
    manager = PERCLOSManager(window_s=2)
    manager.update(True)
    time.sleep(1)
    print(f"PERCLOS after 1s closed: {manager.update(True):.2f}")
    time.sleep(1.5)
    print(f"PERCLOS after window slide: {manager.update(False):.2f}")

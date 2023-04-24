import time
import traceback
from collections import deque
from typing import Optional


class ETAPredictor:
    def __init__(self, lookback_minutes: int = 180):
        self.lookback_seconds = lookback_minutes * 60  # convert minutes to seconds
        self.data = deque()

    def _cleanup_old_data(self):
        current_time = time.time()
        while self.data and current_time - self.data[0][1] > self.lookback_seconds:
            self.data.popleft()

    def predict_eta(
            self, current_step: int, total_steps: int
    ) -> Optional[int]:
        try:
            current_time = time.time()

            # Calculate dynamic log interval based on current logged data
            log_interval = 1
            if len(self.data) > 100:
                log_interval = 10

            # Only log data if last log is at least log_interval seconds ago
            if len(self.data) < 1 or current_time - self.data[-1][1] >= log_interval:
                self.data.append((current_step, current_time))
                self._cleanup_old_data()

            # Only predict if we have enough data
            if len(self.data) < 2 or self.data[-1][1] - self.data[0][1] < 1:
                return None

            first_step, first_time = self.data[0]
            steps_completed = current_step - first_step
            time_elapsed = current_time - first_time

            if steps_completed == 0:
                return None

            time_per_step = time_elapsed / steps_completed
            steps_remaining = total_steps - current_step

            remaining_seconds = steps_remaining * time_per_step
            eta_unix_timestamp = current_time + remaining_seconds

            return int(eta_unix_timestamp)
        except Exception as e:
            print("Error predicting ETA:", e)
            traceback.print_exc()
            return None

    def get_current_speed(self):
        if len(self.data) < 5:
            return None

        last = self.data[-1]
        sample = self.data[-5]
        if len(self.data) > 100:
            sample = self.data[-2]

        steps_completed = last[0] - sample[0]
        time_elapsed = last[1] - sample[1]
        steps_per_second = steps_completed / time_elapsed

        return steps_per_second

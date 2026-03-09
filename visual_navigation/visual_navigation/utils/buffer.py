from builtin_interfaces.msg import Time
from typing import Optional

class MessageBuffer:
    """
    Class for buffering messages with timestamps.
    """
    def __init__(self, max_size: int, wait_for_oldest: bool=False):
        self.max_size = max_size
        self.wait_for_oldest = wait_for_oldest
        self.buffer = []

    def add_msg(self, msg: dict, stamp: Time, time_flt: Optional[float]=None):
        """
        Add a message to the buffer. Save the stamp and the time in float
        for comparison/sorting ops later.
        """
        if time_flt is None:
            time_flt = stamp.sec + stamp.nanosec * 1e-9

        if len(self.buffer) >= self.max_size:
            if self.wait_for_oldest:
                return
            else:
                self.buffer.pop(0)  # Remove the oldest image

        self.buffer.append((msg, stamp, time_flt))

    def get_oldest_msg(self) -> dict:
        """
        Get the oldest message in the buffer.
        """
        if not self.buffer:
            return None
        return self.buffer[0][0]

    def get_closest_msg(self, timestamp: float) -> dict:
        """
        Get the message closest to the given timestamp.
        """
        if not self.buffer:
            return None
        closest_msg = min(self.buffer, key=lambda x: abs(x[2] - timestamp))

        return closest_msg[0]
    
    def pop_oldest_msg(self):
        """
        Remove the oldest message from the buffer.
        """
        if self.buffer:
            return self.buffer.pop(0)

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer = []
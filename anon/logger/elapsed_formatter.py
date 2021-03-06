"""Module contains code for formatting logging output using elapsed time of program"""
import time
from datetime import timedelta


class ElapsedFormatter():
    """Formatter using elapsed time to show how long the process is already running"""

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        """Formats a given record"""
        elapsed_seconds = record.created - self.start_time
        elapsed = timedelta(seconds=elapsed_seconds)
        return "{} | {} | {}".format(record.levelname, elapsed, record.getMessage())

"""TQDM logging handler module. Taken from https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit"""
import logging
import sys
import tqdm

from .elapsed_formatter import ElapsedFormatter


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler used to work with tqdm and stdout"""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.flush = sys.stdout.flush
        self.setFormatter(ElapsedFormatter())

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

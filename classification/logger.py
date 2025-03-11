import sys
import logging


_TIMEZONE = 'Asia/Seoul'

class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self.log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


from __future__ import absolute_import
import os
import sys


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            if not os.path.exists(os.path.dirname(fpath)):
                os.makedirs(os.path.dirname(fpath))
            self.file = open(fpath, 'a')

    def __del__(self):   # when object is deleted, it is automatically called
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):  # when with is ending, it is called
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

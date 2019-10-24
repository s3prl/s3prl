import time
import inspect
from ipdb import set_trace

class Timer:
    def __init__(self):
        self.timings = []
        self.tmp = 0

    def start(self):
        self.tmp = time.time()
    

    def end(self):
        frameinfo = inspect.getouterframes( inspect.currentframe() )[1]
        filename = frameinfo.filename
        filename = '/'.join(filename.split('/')[-2:])
        marker = f'{filename}:{frameinfo.lineno}'
        self.timings.append( (marker, time.time() - self.tmp) )

    def report(self):
        prev = 0
        for items in self.timings:
            for item in items:
                print(item, end="\t")
            print()

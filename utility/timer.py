import time
import inspect
from collections import defaultdict
import torch


class Timer():
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_time = 0

    def start(self):
        self.start_time = time.time()
    

    def end(self):
        frameinfo = inspect.getouterframes( inspect.currentframe() )[1]
        filename = frameinfo.filename
        filename = '/'.join(filename.split('/')[-2:])
        marker = f'{filename}:{frameinfo.lineno}'
        self.timings[marker].append( float(time.time() - self.start_time) )

    def report(self):
        print('[TIMER]:')
        for marker in self.timings:
            print(f'{marker}: {torch.FloatTensor(self.timings[marker]).mean().item()}')

timer = Timer()

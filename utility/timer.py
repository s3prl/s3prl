import time
import inspect
from collections import defaultdict
import torch


class Timer():
    def __init__(self):
        self.timings = {}
        self.start_time = 0

    def start(self):
        self.start_time = time.time()
    

    def end(self):
        frameinfo = inspect.getouterframes( inspect.currentframe() )[1]
        filename = frameinfo.filename
        filename = '/'.join(filename.split('/')[-2:])
        marker = f'{filename}:{frameinfo.lineno}'
        if marker not in self.timings.keys():
            self.timings[marker] = []
        else:
            self.timings[marker].append( float(time.time() - self.start_time) )

    def report(self):
        n_points = len(self.timings.keys())
        if n_points > 0:
            print('[TIMER]:')
            for marker in self.timings:
                print(f'{marker}: {torch.FloatTensor(self.timings[marker]).mean().item()}')
        else:
            print('[TIMER]: No record')

timer = Timer()

import time
import inspect


class Timer():
    def __init__(self):
        self.timings = []
        self.start_time = 0

    def start(self):
        self.start_time = time.time()
    

    def end(self):
        frameinfo = inspect.getouterframes( inspect.currentframe() )[1]
        filename = frameinfo.filename
        filename = '/'.join(filename.split('/')[-2:])
        marker = f'{filename}:{frameinfo.lineno}'
        self.timings.append( (marker, time.time() - self.start_time) )

    def report(self):
        print('[TIMER]:')
        for items in self.timings:
            for item in items:
                print(item, end="\t")
            print()

import numpy as np
import editdistance as ed


def cer(hypothesis, groundtruth, **kwargs):
    er = []
    for p, t in zip(hypothesis, groundtruth):
        er.append(float(ed.eval(p, t)) / len(t))
    return sum(er) / len(er)


def wer(hypothesis, groundtruth, **kwargs):
    er = []
    for p, t in zip(hypothesis, groundtruth):
        p = p.split(' ')
        t = t.split(' ')
        er.append(float(ed.eval(p, t)) / len(t))
    return sum(er) / len(er)

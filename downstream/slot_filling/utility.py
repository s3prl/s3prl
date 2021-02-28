import numpy as np
import editdistance as ed


def cal_er(tokenizer, pred, truth, mode='wer', ctc=False):
    if pred is None:
        return np.nan
    elif len(pred.shape) >= 3:
        pred = pred.argmax(dim=-1)
    er = []

    hypothesis = []
    groundtruth = []
    for p, t in zip(pred, truth):
        p = tokenizer.decode(p.tolist(), ignore_repeat=ctc)
        t = tokenizer.decode(t.tolist())

        hypothesis.append(p)
        groundtruth.append(t)

        if mode == 'wer':
            p = p.split(' ')
            t = t.split(' ')
        er.append(float(ed.eval(p, t))/len(t))

    return sum(er)/len(er), hypothesis, groundtruth

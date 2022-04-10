import editdistance as ed
from sklearn.metrics import accuracy_score


def accuracy(x, y):
    return accuracy_score(x, y)


def ter(hyps, refs):
    error_tokens = 0
    total_tokens = 0
    for h, r in zip(hyps, refs):
        error_tokens += ed.eval(h, r)
        total_tokens += len(r)
    return float(error_tokens) / float(total_tokens)


def wer(hyps, refs):
    hyps = [h.split(" ") for h in hyps]
    refs = [r.split(" ") for r in refs]
    return ter(hyps, refs)


def per(hyps, refs):
    return wer(hyps, refs)


def cer(hyps, refs):
    return ter(hyps, refs)

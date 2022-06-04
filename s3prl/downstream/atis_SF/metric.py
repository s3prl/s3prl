from lib2to3.pgen2 import token
from tokenizers import Tokenizer

SLOT = [i for i in range(3, 83)]
BI_SLOT = [i for i in range(3, 135)]
VALUE = [i for i in range(83, 600+1)]

END_IDX = 1
SEP_IDX = 3
MID_IDX = 4

def parse_entity(seq):
    d = {}
    v = []
    if hasattr(seq, '__iter__'):
        for t in seq: 
            sid = []
            for i, t in enumerate(seq): 
                if t in SLOT+[END_IDX]:
                    sid.append(i)

        for i in range(len(sid)-1): 
            v = seq[sid[i]+1:sid[i+1]]
            if seq[sid[i]] == END_IDX:
                break
            else:
                d[seq[sid[i]]] = v

    return d   


def parse_BI_entity(seq, tokenizer):
    d = {}

    if hasattr(seq, '__iter__'):
        for i, t in enumerate(seq): 
            subword = tokenizer.id_to_token(t)
            if len(subword.split('-')) > 1 and t < 135:
                slot = subword.split('-')[-1]
                if subword.split('-')[0] == 'B':
                    d[slot] = [seq[i-1]]
                elif subword.split('-')[0] == 'I':
                    if slot in d: 
                        d[slot] += [seq[i-1]]
                    else: 
                        continue
            if t == END_IDX:
                break
    return d 

def parse_BIO_entity(seq, tokenizer):
    d = {}
    intent_beg = 0
    if hasattr(seq, '__iter__'):
        # intent 
        if len(seq) > 2:
            intent_beg = seq.pop(0)
            intent_end = seq.pop(-1)

            for i, t in enumerate(seq): 
                subword = tokenizer.id_to_token(t)
                if len(subword.split('-')) > 1 and t < 135:
                    slot = subword.split('-')[-1]
                    if subword.split('-')[0] == 'B':
                        d[slot] = [seq[i-1]]
                    elif subword.split('-')[0] == 'I':
                        if slot in d: 
                            d[slot] += [seq[i-1]]
                        else: 
                            continue
                if t == END_IDX:
                    break
    return d, intent_beg

def parse_split_entity(seq, tokenizer):
    d = {}
    intent = ''
    if hasattr(seq, '__iter__'):
        decode_list = []
        for i, t in enumerate(seq): 
            subword = tokenizer.id_to_token(t)
            decode_list.append(subword)
            if t == END_IDX:
                break
        
        decoded = ''.join(decode_list)
        pairs = decoded.split(';')
        # for intent 
        if len(pairs) > 2: 
            intent = pairs.pop(0)
            pairs.pop(-1)
            # for slot and value
            for p in pairs: 
                value = p.split(':')[0]
                slot = p.split(':')[-1]
                d[slot] = value

    return d, intent

def entity_f1_score(d_gt, d_hyp):
    if len(d_gt.keys()) == 0 and len(d_hyp.keys()) == 0:
        F1 = 1.0
    elif len(d_gt.keys()) == 0:
        F1 = 0.0
    elif len(d_hyp.keys()) == 0:
        F1 = 0.0
    else:
        P, R = 0.0, 0.0
        for slot in d_gt:
            if slot in d_hyp:
                if d_hyp[slot] == d_gt[slot]:
                    R += 1
        R = R / len(d_gt.keys())
        for slot in d_hyp:
            if slot in d_gt:
                if d_hyp[slot] == d_gt[slot]:
                    P += 1
        P = P / len(d_hyp.keys())
        F1 = 2*P*R/(P+R) if (P+R) > 0 else 0.0
    return F1

if __name__ == '__main__':
    seq = 'BALTIMORE B-fromloc.city_name DALLAS B-toloc.city_name ROUND B-round_trip TRIP I-round_trip'
    seq = 'PHILADELPHIA B-toloc.city_name'
    tokenizer = Tokenizer.from_file('/home/daniel094144/data/atis/BI_tokenizer.json')
    id = tokenizer.encode((seq)).ids
    id = [163, 85, 34, 375, 92, 34, 373, 92, 34, 355, 92, 34, 163, 85, 5]
    # print(tokenizer.decode(id))
    d = parse_BI_entity(id, tokenizer)
    print(d)


import editdistance as ed
def uer(hypothesis, groundtruth, **kwargs):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        err += float(ed.eval(p, t))
        tot += len(t)
    return err / tot
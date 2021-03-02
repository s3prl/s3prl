import re
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

def clean(ref):
    ref = re.sub(r'B\-(\S+) ', '', ref)
    ref = re.sub(r' E\-(\S+)', '', ref)
    return ref

def parse(hyp, ref):
    gex = re.compile(r'B\-(\S+) (.+?) E\-\1')

    hyp = re.sub(r' +', ' ', hyp)
    ref = re.sub(r' +', ' ', ref)

    hyp_slots = gex.findall(hyp)
    ref_slots = gex.findall(ref)

    if len(hyp_slots)>0:
        hyp_slots = ';'.join([':'.join([clean(x[1]), x[0]]) for x in hyp_slots])
        ref_slots = ';'.join([':'.join([x[1], x[0]]) for x in ref_slots])
    else:
        hyp_slots = ''
        ref_slots = ''

    ref = clean(ref)
    hyp = clean(hyp)

    return ref, hyp, ref_slots, hyp_slots

def slot_type_f1(hypothesis, groundtruth, **kwargs):
    F1s = []
    for p, t in zip(hypothesis, groundtruth):
        ref_text, hyp_text, ref_slots, hyp_slots = parse(p, t)
        ref_slots = ref_slots.split(';')
        hyp_slots = hyp_slots.split(';')
        unique_slots = []
        ref_dict = {}
        hyp_dict = {}
        if ref_slots[0] != '':
            for ref_slot in ref_slots:
                v, k = ref_slot.split(':')
                ref_dict.setdefault(k, [])
                ref_dict[k].append(v)
        if hyp_slots[0] != '':
            for hyp_slot in hyp_slots:
                v, k = hyp_slot.split(':')
                hyp_dict.setdefault(k, [])
                hyp_dict[k].append(v)
        # Slot Type F1 evaluation
        if len(hyp_dict.keys()) == 0 and len(ref_dict.keys()) == 0:
            F1 = 1.0
        elif len(hyp_dict.keys()) == 0:
            F1 = 0.0
        elif len(ref_dict.keys()) == 0:
            F1 = 0.0
        else:
            P, R = 0.0, 0.0
            for slot in ref_dict:
                if slot in hyp_dict:
                    R += 1
            R = R / len(ref_dict.keys())
            for slot in hyp_dict:
                if slot in ref_dict:
                    P += 1
            P = P / len(hyp_dict.keys())
            F1 = 2*P*R/(P+R) if (P+R) > 0 else 0.0
        F1s.append(F1)
    return sum(F1s) / len(F1s)

def slot_value_cer(hypothesis, groundtruth, **kwargs):
    total_slot = 0
    sf_cer = 0.0
    for p, t in zip(hypothesis, groundtruth):
        ref_text, hyp_text, ref_slots, hyp_slots = parse(p, t)
        ref_slots = ref_slots.split(';')
        hyp_slots = hyp_slots.split(';')
        unique_slots = []
        ref_dict = {}
        hyp_dict = {}
        if ref_slots[0] != '':
            for ref_slot in ref_slots:
                v, k = ref_slot.split(':')
                ref_dict.setdefault(k, [])
                ref_dict[k].append(v)
        if hyp_slots[0] != '':
            for hyp_slot in hyp_slots:
                v, k = hyp_slot.split(':')
                hyp_dict.setdefault(k, [])
                hyp_dict[k].append(v)
        # Slot Value WER/CER evaluation
        unique_slots = list(ref_dict.keys())
        for slot in unique_slots:
            for ref_i, ref_v in enumerate(ref_dict[slot]):
                if slot not in hyp_dict:
                    hyp_v = ''
                    local_cer = cer([hyp_v], [ref_v])
                else:
                    min_cer = 100
                    for hyp_v in hyp_dict[slot]:
                        tmp_cer = cer([hyp_v], [ref_v])
                        if min_cer > tmp_cer:
                            min_cer = tmp_cer
                    local_cer = min_cer
                sf_cer += local_cer
                total_slot += 1

    return sf_cer/total_slot

def slot_value_wer(hypothesis, groundtruth, **kwargs):
    total_slot = 0
    sf_wer = 0.0
    for p, t in zip(hypothesis, groundtruth):
        ref_text, hyp_text, ref_slots, hyp_slots = parse(p, t)
        ref_slots = ref_slots.split(';')
        hyp_slots = hyp_slots.split(';')
        unique_slots = []
        ref_dict = {}
        hyp_dict = {}
        if ref_slots[0] != '':
            for ref_slot in ref_slots:
                v, k = ref_slot.split(':')
                ref_dict.setdefault(k, [])
                ref_dict[k].append(v)
        if hyp_slots[0] != '':
            for hyp_slot in hyp_slots:
                v, k = hyp_slot.split(':')
                hyp_dict.setdefault(k, [])
                hyp_dict[k].append(v)
        # Slot Value WER/CER evaluation
        unique_slots = list(ref_dict.keys())
        for slot in unique_slots:
            for ref_i, ref_v in enumerate(ref_dict[slot]):
                if slot not in hyp_dict:
                    hyp_v = ''
                    local_wer = wer([hyp_v], [ref_v])
                else:
                    min_wer = 100
                    for hyp_v in hyp_dict[slot]:
                        tmp_wer = wer([hyp_v], [ref_v])
                        if min_wer > tmp_wer:
                            min_wer = tmp_wer
                    local_wer = min_wer
                sf_wer += local_wer
                total_slot += 1

    return sf_wer/total_slot

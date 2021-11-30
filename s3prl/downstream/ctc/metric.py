import re
import numpy as np
import editdistance as ed


def cer(hypothesis, groundtruth, **kwargs):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        err += float(ed.eval(p, t))
        tot += len(t)
    return err / tot

def per(*args, **kwargs):
    return wer(*args, **kwargs)

def wer(hypothesis, groundtruth, **kwargs):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = p.split(' ')
        t = t.split(' ')
        err += float(ed.eval(p, t))
        tot += len(t)
    return err / tot

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

    ref_slots = ';'.join([':'.join([x[1], x[0]]) for x in ref_slots])
    if len(hyp_slots)>0:
        hyp_slots = ';'.join([':'.join([clean(x[1]), x[0]]) for x in hyp_slots])
    else:
        hyp_slots = ''

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
    value_hyps = []
    value_refs = []
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
                    value_refs.append(ref_v)
                    value_hyps.append(hyp_v)
                else:
                    min_cer = 100
                    best_hyp_v = ""
                    for hyp_v in hyp_dict[slot]:
                        tmp_cer = cer([hyp_v], [ref_v])
                        if min_cer > tmp_cer:
                            min_cer = tmp_cer
                            best_hyp_v = hyp_v
                    value_refs.append(ref_v)
                    value_hyps.append(best_hyp_v)

    return cer(value_hyps, value_refs)

def slot_value_wer(hypothesis, groundtruth, **kwargs):
    value_hyps = []
    value_refs = []
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
                    value_refs.append(ref_v)
                    value_hyps.append(hyp_v)
                else:
                    min_wer = 100
                    best_hyp_v = ""
                    for hyp_v in hyp_dict[slot]:
                        tmp_wer = wer([hyp_v], [ref_v])
                        if min_wer > tmp_wer:
                            min_wer = tmp_wer
                            best_hyp_v = hyp_v
                    value_refs.append(ref_v)
                    value_hyps.append(best_hyp_v)

    return wer(value_hyps, value_refs)

def slot_edit_f1(hypothesis, groundtruth, loop_over_all_slot, **kwargs):
    test_case, TPs, FNs, FPs = [], 0, 0, 0

    slot2F1  = {} #defaultdict(lambda: [0,0,0]) # TPs, FNs, FPs
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
        # Collecting unique slots
        unique_slots = list(ref_dict.keys())
        if loop_over_all_slot:
            unique_slots += [x for x in hyp_dict if x not in ref_dict]
        # Evaluating slot edit F1
        for slot in unique_slots:
            TP = 0
            FP = 0
            FN = 0
            if slot not in ref_dict: # this never happens in list(ref_dict.keys())
                for hyp_v in hyp_dict[slot]:
                    FP += 1
            else:
                for ref_i, ref_v in enumerate(ref_dict[slot]):
                    if slot not in hyp_dict:
                        FN += 1
                    else:
                        match = False
                        for hyp_v in hyp_dict[slot]:
                            #if ref_i < len(hyp_dict[slot]):
                            #    hyp_v = hyp_dict[slot][ref_i]
                            if hyp_v == ref_v:
                                match = True
                                break
                        if match:
                            TP += 1
                        else:
                            FN += 1
                            FP += 1
            slot2F1.setdefault(slot, [0,0,0])
            slot2F1[slot][0] += TP
            slot2F1[slot][1] += FN
            slot2F1[slot][2] += FP

    all_TPs, all_FNs, all_FPs = 0, 0, 0
    for slot in slot2F1.keys():
        all_TPs += slot2F1[slot][0]
        all_FNs += slot2F1[slot][1]
        all_FPs += slot2F1[slot][2]
    
    return (100.0 * 2*all_TPs/(2*all_TPs + all_FPs + all_FNs))

def slot_edit_f1_full(hypothesis, groundtruth, **kwargs):
    return slot_edit_f1(hypothesis, groundtruth, loop_over_all_slot=True, **kwargs)
 
def slot_edit_f1_part(hypothesis, groundtruth, **kwargs):
    return slot_edit_f1(hypothesis, groundtruth, loop_over_all_slot=False, **kwargs)

import os
import sys
import numpy as np

dir_path = sys.argv[1]
dir_path = os.path.join(dir_path, 'decode_test_clean_fglarge')

best_wer = 99.99
wer_his = []
for (dirpath, dirnames, filenames) in os.walk(dir_path):
    for filename in filenames:
        if 'wer' in filename.split('_')[0]:
            with open(os.path.join(dir_path, filename), 'r') as f:
                wer = float(f.readlines()[1].split(' ')[1])
                wer_his.append(wer)
                if wer < best_wer:
                    best_wer = wer
    break
print('Top 3 lowest wer for the file \'' + dir_path + '\' is :', sorted(wer_his)[:3])
print('Average wer for the file \'' + dir_path + '\' is :', np.mean(wer_his))
# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utility/get_best_score.py ]
#   Synopsis     [ scripts to find the best score from training logs ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


"""
Usage:
    `python utility/get_best_score.py result/downstream/example/log.log dev test +`

This will find the `test` score ranked by the best `dev` score.
+: higher is better
-: lower is better
"""


###############
# IMPORTATION #
###############
import sys


########
# MAIN #
########
def main():

    log_file = str(sys.argv[1])
    
    if len(sys.argv) == 4:
        rank_by = str(sys.argv[2])
        target = str(sys.argv[3])
        large_or_small = str(sys.argv[4])
    else:
        rank_by = 'dev'
        target = 'test'
        large_or_small = '+'

    best_record = [-99999, 0, None]
    if large_or_small == '-': best_record[0] *= -1

    with open(log_file) as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip('\n').split('/')[-1].split('|')
            if len(line) < 3: continue

            prefix = str(line[0].split(':')[-1])
            step = int(line[1].split(':')[-1])
            score = float(line[2].split(':')[-1])

            if rank_by in prefix:
                if compare(score, best_record[0], large_or_small):
                    best_record[0] = score # the score to rank by
                    best_record[1] = step
            
            elif step == best_record[1] and target in prefix:
                best_record[2] = score # the score you want to find
    
    print(f'The best {rank_by} score {best_record[0]} at step {best_record[1]}, accoupanied by this {target} score {best_record[2]}')


def compare(a, b, large_or_small):
    if large_or_small == '+':
        return a > b
    elif large_or_small == '-':
        return a < b
    else:
        raise ValueError(large_or_small)


if __name__ == '__main__':
    main()

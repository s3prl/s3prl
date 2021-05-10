# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utility/get_best_dev.py ]
#   Synopsis     [ script that finds the best dev score from log ]
#   Author       [ Andy T. Liu (github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


"""
Usage:
    `python utility/get_best_dev.py result/downstream/example/log.log`
    `python utility/get_best_dev.py result/downstream/example/log.log $stop_step`
    where $stop_step is an int that specifies the maximum log step to search for.
"""


###############
# IMPORTATION #
###############
import os
import sys
import glob


########
# MAIN #
########
def main():

    log_file = str(sys.argv[1])

    ckpts = glob.glob(os.path.dirname(log_file) + '/states-*.ckpt')
    sorted_ckpts = sorted(ckpts, key=lambda ckpt: int(ckpt.split('.')[0].split('-')[-1]))
    print(f'The last ckpt: {sorted_ckpts[-1]}')

    if len(sys.argv) == 3:
        stop_step = int(sys.argv[2])
    else:
        stop_step = 99999999

    best_dev, best_step, best_test = None, None, None

    with open(log_file) as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip('\n').split(' ')
            
            if line[0].lower() == 'new':
                best_dev = line[-1]
                best_step = line[-2].strip(':')

            if line[0].lower() == 'test':
                if line[-2].strip(':') == best_step:
                    best_test = line[-1]

            if int(line[-2].strip(':')) > stop_step:
                break
    
    print(f'The best dev score {best_dev} at step {best_step}, accoupanied by this test score {best_test}')


if __name__ == '__main__':
    main()

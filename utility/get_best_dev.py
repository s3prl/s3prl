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
    
    print(f'The best dev score {best_dev} at step {best_step}, accoupanied by this test score {best_test}')


if __name__ == '__main__':
    main()

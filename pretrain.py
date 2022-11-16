"""
The testing script for pre-train (temp)

Author
  * Andy T. Liu 2022
"""
import sys

from s3prl.problem.common.pretrain_tera import PretrainTera

PretrainTera().main(sys.argv[1:])


import IPython
import pdb
import numpy as np 

origin = "loop_time_batch.txt"
target = "loop_time.txt"


outer = open(origin,"r").readlines()
inter = open(target,"r").readlines()

# IPython.embed()
# pdb.set_trace()

inter_numbers = []
outer_numbers = []
for i in range(len(outer)):
    float_number = eval(outer[i][-9:-1])
    double_number= eval(inter[i][-9:-1])
    outer_numbers += [float_number]
    inter_numbers += [double_number]

average1 = np.average(inter_numbers)
average2 = np.average(outer_numbers)

print(f"inter = {average1:6.5f}")
print(f"outer = {average2:6.5f}")


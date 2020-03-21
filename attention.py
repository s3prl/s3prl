import torch 
import os 
import wandb
import numpy as np 
import IPython
import pdb 
import matplotlib.pyplot as plt 
import seaborn as sns

target_dir_path = "../result_albert/albert-650000/albert_3l_mask1/mockingjay_libri_sd1337/mockingjayAlbert-490000-dump"

target_dir=os.listdir(target_dir_path)
# wandb.init(project="albert-mockingjay-downstream-task",name="melbase-3-layer")
print(target_dir)
for each_data in target_dir:

    mapping   = torch.load(target_dir_path + "/"+ each_data)
    layer_num = mapping.shape[0]
    heads     = mapping.shape[1]
    print(each_data)
    layer_wise_attention=torch.mean(mapping,dim=1)
    target_attention_path = os.path.join(target_dir_path, each_data+"_directory")
    os.makedirs(target_attention_path)
    
    for layer_num in range(layer_wise_attention.size(0)):

        origin  = layer_wise_attention[layer_num][:20,:20]
        # origin[origin > (mean+2.5*std)] = mean+2.5*std
        # origin[origin < (mean-2.5*std)] = mean-2.5*std
        sns.heatmap(origin, cmap=plt.cm.spring)
        plt.savefig(target_attention_path + "/" + str(layer_num) + ".png")
        plt.clf()
        target_attention_path_each_head = os.path.join(target_attention_path , "layer_" + str(layer_num)+"_head")
        os.makedirs(target_attention_path_each_head)
        for index in range(12):
            origin = mapping[layer_num,index][:20,:20]
            sns.heatmap(origin, cmap=plt.cm.gist_heat)
            plt.savefig(target_attention_path_each_head + "/" + str(index) + ".png")
            plt.clf()
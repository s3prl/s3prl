#!/bin/bash

gpus=$1
batch=2
distributed="-m torch.distributed.launch --nproc_per_node ${gpus}"
upstream="hubert_base"
tasks="pr,ks,ic,sid,er,asr,sf,sv,sd"

#simple_tasks="ks,ic,sid,er,asr,sf,sv,sd"
#hard_tasks="pr"
#ckpt_path="/SpeechNet/s3prl/result/downstream/pretrained-downstream_finetune-hubert_base-weighted-sum-of-layers_ks,ic,sid,er,asr,sf,sv,sd_batch-1_gpus-2/states-200000.ckpt"
#simple_tasks="pr,ic,sid,er,asr,sf,sv,sd"
#hard_tasks="ks"
#ckpt_path="/SpeechNet/s3prl/result/downstream/pretrained-downstream_finetune-hubert_base-weighted-sum-of-layers_pr,ic,sid,er,asr,sf,sv,sd_batch-1_gpus-2/states-200000.ckpt"
#simple_tasks="pr,ks,sid,er,asr,sf,sv,sd"
#hard_tasks="ic"
#ckpt_path="/SpeechNet/s3prl/result/downstream/pretrained-downstream_finetune-hubert_base-weighted-sum-of-layers_pr,ks,sid,er,asr,sf,sv,sd_batch-1_gpus-2/states-200000.ckpt"
#simple_tasks="pr,ks,ic,er,asr,sf,sv,sd"
#hard_tasks="sid"
#ckpt_path="/SpeechNet/s3prl/result/downstream/pretrained-downstream_finetune-hubert_base-weighted-sum-of-layers_pr,ks,ic,er,asr,sf,sv,sd_batch-1_gpus-2/states-200000.ckpt"
#simple_tasks="pr,ks,ic,sid,asr,sf,sv,sd"
#hard_tasks="er"
#ckpt_path="/SpeechNet/s3prl/result/downstream/pretrained-downstream_finetune-hubert_base-weighted-sum-of-layers_pr,ks,ic,sid,asr,sf,sv,sd_batch-1_gpus-2/states-200000.ckpt"
#simple_tasks="pr,ks,ic,sid,er,sf,sv,sd"
#hard_tasks="asr"
#ckpt_path="/SpeechNet/s3prl/result/downstream/pretrained-downstream_finetune-hubert_base-weighted-sum-of-layers_pr,ks,ic,sid,er,sf,sv,sd_batch-1_gpus-2/states-200000.ckpt"
#simple_tasks="pr,ks,ic,sid,er,asr,sv,sd"
#hard_tasks="sf"
#ckpt_path="/SpeechNet/s3prl/result/downstream/pretrained-downstream_finetune-hubert_base-weighted-sum-of-layers_pr,ks,ic,sid,er,asr,sv,sd_batch-1_gpus-2/states-200000.ckpt"
#simple_tasks="pr,ks,ic,sid,er,asr,sf,sd"
#hard_tasks="sv"
#ckpt_path="/SpeechNet/s3prl/result/downstream/pretrained-downstream_finetune-hubert_base-weighted-sum-of-layers_pr,ks,ic,sid,er,asr,sf,sd_batch-1_gpus-2/states-200000.ckpt"
simple_tasks="pr,ks,ic,sid,er,asr,sf,sv"
hard_tasks="sd"
ckpt_path="/SpeechNet/s3prl/result/downstream/pretrained-downstream_finetune-hubert_base-weighted-sum-of-layers_pr,ks,ic,sid,er,asr,sf,sv_batch-1_gpus-2/states-200000.ckpt"

# simple_tasks="pr,ks,ic,sid,er"
# hard_tasks="asr,sf,sv,sd"

# no distributed
# python3 run_downstream.py -m train -n train_all -u hubert_base -d pr,ks,ic,sid,er,asr,sf,sv,sd -c downstream/multitask.yaml -s default -f

# all (last layer) + pretrained all downstream tasks
# NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n pretrained-downstream_finetune-${upstream}_${tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $tasks -c downstream/multitask.yaml -s default -f --init_ckpt /SpeechNet/checkpoints

# all (weighted sum of layers) + pretrained all downtream tasks
# NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n pretrained-downstream_finetune-${upstream}-weighted-sum-of-layers_${tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $tasks -c downstream/multitask.yaml -s layer_results -f --init_ckpt /SpeechNet/checkpoints

# all (weighted sum of layers) + pretrained simple downtream tasks
# NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n pretrained-downstream_finetune-${upstream}-weighted-sum-of-layers_${simple_tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $simple_tasks -c downstream/multitask.yaml -s layer_results -f --init_ckpt /SpeechNet/checkpoints

# all (weighted sum of layers) on only hard tasks + pretrained upstream from a previous experiment (simple)
NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n pretrained-downstream_${simple_tasks}-${upstream}-weighted-sum-of-layers_${hard_tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $hard_tasks -c downstream/multitask.yaml -s layer_results --init_ckpt /SpeechNet/checkpoints --init_upstream_ckpt $ckpt_path

# ====== autoloss or pcgrad ===============================================================================

# all (weighted sum of layers) + pretrained all downtream tasks + auto_loss_weights + pcgrad
# NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n autoloss_pcgrad_pretrained-downstream_finetune-${upstream}-weighted-sum-of-layers_${tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $tasks -c downstream/multitask.yaml -s layer_results -f --init_ckpt /SpeechNet/checkpoints --auto_loss_weights --pcgrad

# all (weighted sum of layers) + pretrained all downtream tasks + auto_loss_weights
# NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n autoloss_pretrained-downstream_finetune-${upstream}-weighted-sum-of-layers_${tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $tasks -c downstream/multitask.yaml -s layer_results -f --init_ckpt /SpeechNet/checkpoints --auto_loss_weights

# all (weighted sum of layers) + pretrained all downtream tasks + pcgrad
# NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n pcgrad_pretrained-downstream_finetune-${upstream}-weighted-sum-of-layers_${tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $tasks -c downstream/multitask.yaml -s layer_results -f --init_ckpt /SpeechNet/checkpoints --pcgrad

# ====== simple tasks autoloss or pcgrad ===============================================================================

# all (weighted sum of layers) + pretrained five simple downtream tasks (pr,ks,sid,ic,er) + auto_loss_weights + pcgrad
# NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n autoloss_pcgrad_pretrained-downstream_finetune-${upstream}-weighted-sum-of-layers_${simple_tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $simple_tasks -c downstream/multitask.yaml -s layer_results -f --init_ckpt /SpeechNet/checkpoints --auto_loss_weights --pcgrad

# all (weighted sum of layers) + pretrained five simple downtream tasks (pr,ks,sid,ic,er) + auto_loss_weights
# NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n autoloss_pretrained-downstream_finetune-${upstream}-weighted-sum-of-layers_${simple_tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $simple_tasks -c downstream/multitask.yaml -s layer_results -f --init_ckpt /SpeechNet/checkpoints --auto_loss_weights

# all (weighted sum of layers) + pretrained five simple downtream tasks (pr,ks,sid,ic,er) + pcgrad
# NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n pcgrad_pretrained-downstream_finetune-${upstream}-weighted-sum-of-layers_${simple_tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $simple_tasks -c downstream/multitask.yaml -s layer_results -f --init_ckpt /SpeechNet/checkpoints --pcgrad

# ====== hard tasks (no finetune) pretrained autoloss or pcgrad ==========================================================

# all (weighted sum of layers) on only hard tasks (asr,sf,sv,sd) + pretrained upstream from a previous experiment (simple) + auto_loss_weights
# NCCL_ASYNC_ERROR_HANDLING=1 python3 $distributed run_downstream.py -m train -n autoloss_pretrained-downstream_simple-${upstream}-weighted-sum-of-layers_${hard_tasks}_batch-${batch}_gpus-${gpus} -u $upstream -d $hard_tasks -c downstream/multitask.yaml -s layer_results --init_ckpt /SpeechNet/checkpoints --init_upstream_ckpt /SpeechNet/s3prl/result/downstream/autoloss_pretrained-downstream_finetune-hubert_base-weighted-sum-of-layers_pr,ks,ic,sid,er_batch-1_gpus-2/states-200000.ckpt

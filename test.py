# import fairseq

# ckpt_path = "hubert_pt/hubert_base_ls960.pt"
# models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path], strict=False)
# model = models[0]
# print(model)


import torch
print(torch.hub.list('s3prl/s3prl'))

# import torch


# t = torch.ones(1,1,28,28)
# t = t.cuda()
# print(t)
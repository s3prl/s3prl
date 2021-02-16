import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

'''
new Augment function
'''
class Augment(nn.Module):
    def __init__(self, T=40, num_masks=1, replace_with_zero=False, F=27):#ori: T = 40
        super(Augment, self).__init__()    
        self.T=T
        self.num_masks=num_masks
        self.replace_with_zero=replace_with_zero
        self.F=F
        self.spec=None
    #@torch.jit.script_method
    def forward(self, spec):
        spec = spec.permute(1, 0)

        spec = self.time_mask(spec, T=self.T, num_masks=self.num_masks, replace_with_zero=self.replace_with_zero)
        spec = self.freq_mask(spec, F=self.F, num_masks=self.num_masks, replace_with_zero=self.replace_with_zero)
        spec = spec.permute(1, 0)
        
        return spec
    def normalize(self, spec):
        spec = (spec-spec.mean())/spec.std()
        return spec        

    def time_mask(self, spec, T=100, num_masks=1, replace_with_zero=False):
        cloned = spec
        len_spectro = cloned.shape[1]
        
        for i in range(0, num_masks):
            t = torch.randint(0, self.T, (1,)).item()
            t_zero = torch.randint(0, len_spectro-t, (1,)).item()
            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned
            mask_end = torch.randint(t_zero, t_zero+t, (1, )).item()

            if (replace_with_zero): cloned[:,t_zero:mask_end] = 0
            else: cloned[:,t_zero:mask_end] = cloned.mean()
        return cloned

    def freq_mask(self, spec, F=27, num_masks=1, replace_with_zero=False):
        cloned = spec
        num_mel_channels = cloned.shape[0]

        for i in range(0, num_masks):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            if (replace_with_zero): cloned[f_zero:mask_end, :] = 0
            else: cloned[f_zero:mask_end, :] = cloned.mean()

        return cloned

    def tensor_to_img(self, spectrogram, filename):
        spectrogram = self.to_db_scale(spectrogram)
        spectrogram = spectrogram.detach().numpy()
        #print(spectrogram[0])
        plt.figure(figsize=(14,1)) # arbitrary, looks good on my screen.
        plt.imshow(spectrogram)
        plt.savefig(filename)
        #plt.show()
        #display(spectrogram.shape)
    def to_db_scale(self, spectrogram):
        db_sp = 20*torch.log10(spectrogram)
        return db_sp

    def time_warp(self, spec, W=5):
        num_rows = spec.shape[1]
        spec_len = spec.shape[2]
        device = spec.device

        y = num_rows//2
        horizontal_line_at_ctr = spec[0][y]
        assert len(horizontal_line_at_ctr) == spec_len

        point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
        assert isinstance(point_to_warp, torch.Tensor)

        # Uniform distribution from (0,W) with chance to be up to W negative
        dist_to_warp = random.randrange(-W, W)
        src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                            torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
        warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
        return warped_spectro.squeeze(3)



class TrainableAugment(nn.Module): 
    def __init__(self, aug_type, block_config, optimizer):
        super(TrainableAugment, self).__init__()
        self.aug_type = aug_type # 1: train aug 2: use pretrain aug module 3:previous specaug 4: no aug(only means no aug on spectrogram)
        if self.aug_type == 3:
            self.aug_model = Augment(**block_config)
            self.optimizer = None
        elif self.aug_type == 1:
            self.aug_model = _TrainableAugmentModel(**block_config)
            self.aug_model.set_trainable_aug()
            opt = getattr(torch.optim, optimizer.pop('optimizer', 'sgd'))
            self.optimizer = opt(self.aug_model.parameters(), **optimizer)
        elif self.aug_type == 2:
            self.aug_model = _TrainableAugmentModel(**block_config)
            self.aug_model.disable_trainable_aug()
            opt = getattr(torch.optim, optimizer.pop('optimizer', 'sgd'))
            self.optimizer = opt(self.aug_model.parameters(), **optimizer)
            
        else:
            raise NotImplementedError

    def get_new_noise(self, feat):
        return self.aug_model.get_new_noise(feat)

    def forward(self, feat, feat_len=None, noise=None):
        if self.aug_type == 1 or self.aug_type == 2:
            feat = self.aug_model(feat, feat_len, noise=noise)
        if self.aug_type == 3:
            feat = self.aug_model(feat)
        return feat

    def step(self):
        self.optimizer.step()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def load_ckpt(self, ckpt_path):
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path)
            self.aug_model.load_state_dict(ckpt['aug_model'])
            self.optimizer.load_state_dict(ckpt['aug_optimizer'])

    def save_ckpt(self, ckpt_path):
        full_dict = {
            "aug_model": self.aug_model.state_dict(),
            "aug_optimizer": self.optimizer.state_dict()
        }
        torch.save(full_dict, ckpt_path)

class _TrainableAugmentModel(nn.Module):
    def __init__(self, T_num_masks=1, F_num_masks=1, noise_dim=None, dim=[10, 10, 10], replace_with_zero=False, width_init_bias=-3.):
        '''
        noise_dim: the input noise to the generation network
        '''
        super(_TrainableAugmentModel, self).__init__()
        self.T_num_masks = T_num_masks
        self.F_num_masks = F_num_masks

        self.output_dim = (self.T_num_masks+self.F_num_masks)*2 # position and width
        if noise_dim is None:
            self.noise_dim = self.output_dim
        else:
            self.noise_dim = noise_dim

        self.dim = dim
        self.replace_with_zero = replace_with_zero

        self.trainable_aug = True # whether this module trainable, or serve as a normal augmentation module

        module_list = []
        prev_dim = self.noise_dim
        for d in self.dim:
            module_list.append(nn.Linear(prev_dim, d))
            prev_dim = d
            module_list.append(nn.ReLU())
        module_list.append(nn.Linear(prev_dim, self.output_dim))
        module_list.append(nn.Sigmoid())

        # init last linear 
        # aug_param layout [T_mask_center, T_mask_width, F_mask_center, F_mask_width] 
        # init mask width to be small 
        module_list[-2].bias[self.T_num_masks:2*self.T_num_masks].data.fill_(width_init_bias)
        module_list[-2].bias[(2*self.T_num_masks+self.F_num_masks):(2*self.T_num_masks+2*self.F_num_masks)].data.fill_(width_init_bias)
        self.layers = nn.Sequential(*module_list)

    def set_trainable_aug(self):
        self.trainable_aug = True

    def disable_trainable_aug(self):
        self.trainable_aug = False

    def forward(self, feat, feat_len, noise=None):
        filling_value = 0. if self.replace_with_zero else self._get_mask_mean(feat, feat_len)
        # print('filling_value', filling_value)
        aug_param = self._generate_aug_param(feat, noise=noise)
        # print('aug_param', aug_param)

        if self.trainable_aug:
            return self._forward_trainable(feat, feat_len, filling_value, aug_param)
        else:
            return self._forward_not_trainable(feat, feat_len, filling_value, aug_param)
    
    def get_new_noise(self, feat):
        return torch.randn(feat.shape[0], self.noise_dim).to(feat.device)

    def _generate_aug_param(self, feat, noise=None):
        if noise is None:
            noise = self.get_new_noise(feat)
        aug_param = self.layers(noise)
        return aug_param # [B, 2*(T_num_masks+F_num_masks)]

    def _forward_trainable(self, feat, feat_len, filling_value, aug_param):
        '''
        filling_value: [B]
        aug_param: [B, 2*(F_num_mask+T_num_mask)]
        '''
        def _get_soft_log_left_weight(mask_center, mask_width, length, max_len):
            '''
            length: [B] # can be T or F dimension
            mask_center: [B, num_mask]
            mask_width: [B, num_mask]
            output: [B, l]
            '''
            SIGMOID_THRESHOLD = 5 # assume 2*SIGMOID_THRESHOLD is the width, because sigmoid(SIGMOID_THRESHOLD) starts to very close to 1
            device = mask_center.device

            position = torch.div(torch.arange(start=0, end=max_len, device=device).unsqueeze(0), length.unsqueeze(1)) # [B, l]

            dist_to_center = mask_center.unsqueeze(-1) - position.unsqueeze(1) # [B, num_mask, l]
            abs_ratio = torch.abs(dist_to_center)/mask_width.unsqueeze(-1) # [B, num_mask, l]

            log_mask_weight = F.logsigmoid((abs_ratio*(2*SIGMOID_THRESHOLD))-SIGMOID_THRESHOLD) # [B, num_mask, l]
            log_mask_weight = log_mask_weight.sum(1) # [B, l]
            return log_mask_weight

        # mask T
        T_log_mask_weight = _get_soft_log_left_weight(aug_param[:, :self.T_num_masks], \
                                                      aug_param[:, self.T_num_masks:2*self.T_num_masks], \
                                                      feat_len, feat.shape[1])
        # mask F
        F_log_mask_weight = _get_soft_log_left_weight(aug_param[:, 2*self.T_num_masks:2*self.T_num_masks+self.F_num_masks], \
                                                      aug_param[:, 2*self.T_num_masks+self.F_num_masks:2*self.T_num_masks+2*self.F_num_masks], \
                                                      feat_len.new_tensor([feat.shape[-1]]), feat.shape[-1])
        total_log_left_weight = T_log_mask_weight.unsqueeze(-1)+F_log_mask_weight.unsqueeze(1) # [B, T, F]
        total_left_weight = torch.exp(total_log_left_weight)
        # print(total_left_weight)

        new_feat = feat*total_left_weight + filling_value.unsqueeze(1).unsqueeze(1)*(1-total_left_weight)
        # we do not mask the fill result, cuz assume ASR model will handle this
        return new_feat

    def _forward_not_trainable(self, feat, feat_len, filling_value, aug_param):
        # use fix 
        def _get_hard_mask(mask_center, mask_width, length, max_len):
            '''
            length: [B] # can be T or F dimension
            mask_center: [B, num_mask]
            mask_width: [B, num_mask]
            output: [B, l]
            '''
            device = mask_center.device
            position = torch.div(torch.arange(start=0, end=max_len, device=device).unsqueeze(0), length.unsqueeze(1)) # [B, l]

            mask_left = mask_center  - mask_width/2 - torch.rand_like(mask_center)/length.unsqueeze(-1)
            mask_right = mask_center + mask_width/2 + torch.rand_like(mask_center)/length.unsqueeze(-1)

            mask_left  = position.unsqueeze(1) > mask_left.unsqueeze(-1) # [B, num_mask, l]
            mask_right = position.unsqueeze(1) < mask_right.unsqueeze(-1)
            mask = mask_left & mask_right
            mask = mask.any(1)

            return mask

        # mask T
        T_mask = _get_hard_mask(aug_param[:, :self.T_num_masks], \
                                aug_param[:, self.T_num_masks:2*self.T_num_masks], \
                                feat_len, feat.shape[1])
        # mask F
        F_mask = _get_hard_mask(aug_param[:, 2*self.T_num_masks:2*self.T_num_masks+self.F_num_masks], \
                                aug_param[:, 2*self.T_num_masks+self.F_num_masks:2*self.T_num_masks+2*self.F_num_masks], \
                                feat_len.new_tensor([feat.shape[-1]]), feat.shape[-1])

        total_mask = T_mask.unsqueeze(-1) | F_mask.unsqueeze(1)
        # print(total_mask)

        new_feat = torch.where(total_mask, filling_value.unsqueeze(1).unsqueeze(1), feat)
        # we do not mask the fill result, cuz assume ASR model will handle this
        return new_feat    

    def _mask_with_length(self, feat, feat_len, fill_num=0.):
        '''
        mask the second dimensin of the feat with feat_len
        feat: rank>=2 [B x L x ...]
        feat_len: [B]
        '''
        assert (len(feat.size()) >= 2)
        assert (feat.size()[0] == feat_len.size()[0])

        max_len = feat.size()[1]
        rank = len(feat.size())

        a = torch.arange(max_len).unsqueeze(0).int()
        b = feat_len.unsqueeze(1).int()
        if feat.is_cuda:
            a = a.cuda()
            b = b.cuda()

        mask = torch.ge(a, b)
        mask = mask.view(mask.size()[0],mask.size()[1], *([1]*(rank-2))).expand_as(feat) #mask: where to pad zero
        #if feat.is_cuda:
            #mask = mask.cuda()
        feat_clone = feat.clone() # prevent inplace substitution
        feat_clone[mask] = fill_num
        return feat_clone, ~mask

    def _get_mask_mean(self, feat, feat_len):
        feat, mask = self._mask_with_length(feat, feat_len)
        return feat.sum([1,2])/mask.float().sum([1,2])

if __name__ == '__main__':
    trainable_aug = _TrainableAugmentModel().cuda()

    batch_size = 3
    l = 20
    feat_dim = 5
    feat = torch.rand(batch_size, l, feat_dim).cuda()
    # feat_len = torch.randint(1, l-1, size=(batch_size,)).cuda()
    feat_len = torch.tensor([l]*batch_size).cuda()

    opt = torch.optim.SGD(trainable_aug.parameters(), lr=1)
    new_feat_soft  = trainable_aug(feat, feat_len)

    trainable_aug.disable_trainable_aug()
    new_feat_hard  = trainable_aug(feat, feat_len)
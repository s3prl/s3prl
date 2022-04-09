import torch
import torch.nn as nn
import torch.nn.functional as F
from s3prl import Output

from . import NNModule

class softmax(NNModule):
    def __init__(
                    self, 
                    input_size: int, 
                    output_size: int 
                ):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.criertion = nn.CrossEntropyLoss()

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x, label):
        """
        Args:
            x (torch.Tensor): (batch_size, input_size)
            label (torch.LongTensor): (batch_size, )

        Return:
            loss (torch.float)
            logit (torch.Tensor): (batch_size, )
        """
        
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.input_size

        x    = F.normalize(x, dim=1)
        x    = self.fc(x)
        loss = self.criertion(x, label)
        
        return Output(loss=loss, logit=x)

class amsoftmax(NNModule):
    def __init__(
                    self, 
                    input_size: int, 
                    output_size: int, 
                    margin: float = 0.2, 
                    scale: float = 30
                ):
        super(amsoftmax, self).__init__()
        
        self.W = torch.nn.Parameter(torch.randn(input_size, output_size), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x, label):
        """
        Args:
            x (torch.Tensor): (batch_size, input_size)
            label (torch.LongTensor): (batch_size, )

        Return:
            loss (torch.float)
            logit (torch.Tensor): (batch_size, )
        """

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.input_size

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.arguments.margin)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.arguments.scale * costh_m
        loss = self.ce(costh_m_s, label)

        return Output(loss=loss, logit=costh_m_s)
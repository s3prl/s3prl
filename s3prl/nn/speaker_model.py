from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from s3prl import Output

from . import NNModule
from .pooling import Self_Attentive_Pooling, Attentive_Statistics_Pooling, Temporal_Average_Pooling, Temporal_Statistics_Pooling

class TDNN(NNModule):
        
    def __init__(
                    self, 
                    input_size: int, 
                    output_size: int,
                    context_size: int,
                    dilation: int,
                    dropout_p: float = 0.0,
                    stride: int = 1,
                    batch_norm: bool = True,
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super().__init__()
        
        self.kernel = nn.Linear(input_size*context_size, output_size)
        self.nonlinearity = nn.ReLU()
        if self.arguments.batch_norm:
            self.bn = nn.BatchNorm1d(output_size)
        if self.arguments.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size
    
    def forward(self, x):
        '''
        input: 
            x: with size (batch, seq_len, input_size)
        output: 
            x: with size (batch, seq_len, output_size)
        '''

        _, _, d = x.shape
        assert (d == self.input_size), 'Input size was wrong. Expected ({}), got ({})'.format(self.input_size, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.arguments.context_size, self.input_size), 
                        stride=(1, self.input_size), 
                        dilation=(self.arguments.dilation,1)
                    )

        x = x.transpose(1,2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        
        if self.arguments.dropout_p:
            x = self.drop(x)

        if self.arguments.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x

class XVector(NNModule):
    def __init__(
                    self, 
                    input_size: int,
                    output_size: int = 1500,
                    **kwargs
                ):
        super().__init__()
        '''
        XVector model as in https://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
        This model only include the blocks before the pooling layer
        '''
        self.module = nn.Sequential(
            TDNN(input_size=input_size, output_size=512, context_size=5, dilation=1),
            TDNN(input_size=512, output_size=512, context_size=3, dilation=2),
            TDNN(input_size=512, output_size=512, context_size=3, dilation=3),
            TDNN(input_size=512, output_size=512, context_size=1, dilation=1),
            TDNN(input_size=512, output_size=output_size, context_size=1, dilation=1),
        )

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size
    
    def forward(self, x):
        '''
        input: 
            x: size (batch, seq_len, input_size)
        output:
            x: size (batch, seq_len, output_size)
        '''

        x = self.module(x)

        return Output(output=x)


class speaker_embedding_extractor(NNModule):
    def __init__(
                    self,
                    backbone: str,
                    pooling_type: str,
                    input_size: int,
                    output_size: int
                ):
        super().__init__()

        if self.arguments.backbone == "XVector":
            self.backbone = XVector(
                input_size=input_size, output_size=output_size
            )
        else:
            raise ValueError('{} backbone type is not defined'.format(args.backbone))

        pooling_type = self.arguments.pooling_type
        if pooling_type == "Temporal_Average_Pooling" or pooling_type == "TAP":
            self.pooling = Temporal_Average_Pooling(
                input_size=self.backbone.output_size, output_size=self.backbone.output_size
            )
            self.fc = nn.Linear(self.backbone.output_size, self.backbone.output_size)

        elif pooling_type == "Temporal_Statistics_Pooling" or pooling_type == "TSP":
            self.pooling = Temporal_Statistics_Pooling(
                input_size=self.backbone.output_size, output_size=self.backbone.output_size*2
            )
            self.fc = nn.Linear(self.backbone.output_size*2, self.backbone.output_size)

        elif pooling_type == "Self_Attentive_Pooling" or pooling_type == "SAP":
            self.pooling = Self_Attentive_Pooling(
                input_size=self.backbone.output_size, output_size=self.backbone.output_size
            )
            self.fc = nn.Linear(self.backbone.output_size, self.backbone.output_size)

        elif pooling_type == "Attentive_Statistics_Pooling" or pooling_type == "ASP":
            self.pooling = Attentive_Statistics_Pooling(
                input_size=self.backbone.output_size, output_size=self.backbone.output_size*2
            )
            self.fc = nn.Linear(self.backbone.output_size*2, self.backbone.output_size)

        else:
            raise ValueError('{} pooling type is not defined'.format(pooling_type))

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, x, xlen=None):
        '''
        input: 
            x: size (batch, seq_len, input_size)
        output:
            x: size (batch, output_size)
        '''

        x = self.backbone(x).slice(1)
        x = x.permute(0, 2, 1)
        # TODO: add xlen into pooling
        x = self.pooling(x).slice(1)
        x = self.fc(x)

        return Output(output=x)
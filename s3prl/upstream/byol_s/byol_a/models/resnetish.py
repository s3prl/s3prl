"""Resnet-ish adaptation for Pytorch

Implementation largely based on https://github.com/daisukelab/sound-clf-pytorch

Original paper:
@inproceedings{hershey2017cnn,
  title={CNN architectures for large-scale audio classification},
  author={Hershey, Shawn and Chaudhuri, Sourish and Ellis, Daniel PW and Gemmeke, Jort F and Jansen, Aren and Moore, R Channing and Plakal, Manoj and Platt, Devin and Saurous, Rif A and Seybold, Bryan and others},
  booktitle={2017 ieee international conference on acoustics, speech and signal processing (icassp)},
  pages={131--135},
  year={2017},
  organization={IEEE}
}
"""

from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import weight_norm


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    standardize_weights: bool = False,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
    return weight_norm(conv) if standardize_weights else conv


def conv1x1(
    in_planes: int, out_planes: int, stride: int = 1, standardize_weights: bool = False
) -> nn.Conv2d:
    """1x1 convolution"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return weight_norm(conv) if standardize_weights else conv


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        standardize_weights: bool = False,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(
            inplanes, planes, stride, standardize_weights=standardize_weights
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, standardize_weights=standardize_weights)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        standardize_weights: bool = False,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, standardize_weights=standardize_weights)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(
            width,
            width,
            stride,
            groups,
            dilation,
            standardize_weights=standardize_weights,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(
            width, planes * self.expansion, standardize_weights=standardize_weights
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetish(nn.Module):
    # sample rate and embedding sizes are required model attributes for the HEAR API
    sample_rate = 16000
    embedding_size = 2048
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        standardize_weights: bool = False,
    ) -> None:
        super(ResNetish, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.conv1 = weight_norm(conv1) if standardize_weights else conv1
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], standardize_weights=standardize_weights
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            standardize_weights=standardize_weights,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            standardize_weights=standardize_weights,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            standardize_weights=standardize_weights,
        )
        # self.avgpool = nn.AvgPool2d((4, 6))
        # self.fc = nn.Linear(512 * 24 * block.expansion, num_classes)
        # self.flatten = nn.Flatten()
        # self.add_max_mean = Lambda(lambda x: x.mean(1) + x.amax(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        standardize_weights: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes, planes * block.expansion, stride, standardize_weights
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # BYOL-A max-mean operation
        x = x.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C * D))  # (batch, time, mel*ch)

        x = x.mean(1) + x.amax(1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnetish(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNetish:
    model = ResNetish(block, layers, **kwargs)
    # if pretrained:
    # state_dict = torch.utils.load_state_dict_from_url(model_urls[arch], progress=progress)
    # model.load_state_dict(state_dict)
    return model


def resnetish10(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNetish:
    r"""ResNet-10 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnetish(
        "resnetish18", BasicBlock, [1, 1, 1, 1], pretrained, progress, **kwargs
    )


def resnetish18(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNetish:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnetish(
        "resnetish18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def resnetish34(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNetish:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Adapted for Audio from
    `"CNN architectures for large-scale audio classification" <https://arxiv.org/abs/1609.09430>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnetish(
        "resnetish34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnetish50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNetish:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Adapted for Audio from
    `"CNN architectures for large-scale audio classification" <https://arxiv.org/abs/1609.09430>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnetish(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/cpc/feature_loader.py ]
#   Synopsis     [ the cpc feature loader ]
#   Author       [ Facebook Research ]
#   Copyright    [ Facebook, Inc. and its affiliates ]
#   Reference    [ https://github.com/facebookresearch/CPC_audio ]
"""*********************************************************************************************"""


import argparse
import json

###############
# IMPORTATION #
###############
import os

# -------------#
import torch
import torchaudio

# -------------#
from .cpc_default_config import get_default_cpc_config
from .model import ConcatenatedModel, CPCModel


class FeatureModule(torch.nn.Module):
    r"""
    A simpler interface to handle CPC models. Useful for a smooth workflow when
    working with CPC trained features.
    """

    def __init__(self, featureMaker, get_encoded, collapse=False):
        super(FeatureModule, self).__init__()
        self.get_encoded = get_encoded
        self.featureMaker = featureMaker
        self.collapse = collapse

    def getDownsamplingFactor(self):
        return self.featureMaker.gEncoder.DOWNSAMPLING

    def forward(self, data):

        batchAudio, label = data
        cFeature, encoded, _ = self.featureMaker(batchAudio.cuda(), label)
        if self.get_encoded:
            cFeature = encoded
        if self.collapse:
            cFeature = cFeature.contiguous().view(-1, cFeature.size(2))
        return cFeature


class ModelPhoneCombined(torch.nn.Module):
    r"""
    Concatenates a CPC feature maker and a phone predictor.
    """

    def __init__(self, model, criterion, oneHot):
        r"""
        Arguments:
            model (FeatureModule): feature maker
            criterion (PhoneCriterion): phone predictor
            oneHot (bool): set to True to get a one hot output
        """
        super(ModelPhoneCombined, self).__init__()
        self.model = model
        self.criterion = criterion
        self.oneHot = oneHot

    def getDownsamplingFactor(self):
        return self.model.getDownsamplingFactor()

    def forward(self, data):
        c_feature = self.model(data)
        pred = self.criterion.getPrediction(c_feature)
        P = pred.size(2)

        if self.oneHot:
            pred = pred.argmax(dim=2)
            pred = toOneHot(pred, P)
        else:
            pred = torch.nn.functional.softmax(pred, dim=2)
        return pred


def loadArgs(args, locArgs, forbiddenAttr=None):
    for k, v in vars(locArgs).items():
        if forbiddenAttr is not None:
            if k not in forbiddenAttr:
                setattr(args, k, v)
        else:
            setattr(args, k, v)


def getCheckpointData(pathDir):
    if not os.path.isdir(pathDir):
        return None
    checkpoints = [
        x
        for x in os.listdir(pathDir)
        if os.path.splitext(x)[1] == ".pt" and os.path.splitext(x[11:])[0].isdigit()
    ]
    if len(checkpoints) == 0:
        print("No checkpoints found at " + pathDir)
        return None
    checkpoints.sort(key=lambda x: int(os.path.splitext(x[11:])[0]))
    data = os.path.join(pathDir, checkpoints[-1])
    with open(os.path.join(pathDir, "checkpoint_logs.json"), "rb") as file:
        logs = json.load(file)

    with open(os.path.join(pathDir, "checkpoint_args.json"), "rb") as file:
        args = json.load(file)

    args = argparse.Namespace(**args)
    defaultArgs = get_default_cpc_config()
    loadArgs(defaultArgs, args)

    return os.path.abspath(data), logs, defaultArgs


def getEncoder(args):

    if args.encoder_type == "mfcc":
        from .model import MFCCEncoder

        return MFCCEncoder(args.hiddenEncoder)
    elif args.encoder_type == "lfb":
        from .model import LFBEnconder

        return LFBEnconder(args.hiddenEncoder)
    else:
        from .model import CPCEncoder

        return CPCEncoder(args.hiddenEncoder, args.normMode)


def getAR(args):
    if args.arMode == "transformer":
        from .transformers import buildTransformerAR

        arNet = buildTransformerAR(
            args.hiddenEncoder, 1, args.sizeWindow // 160, args.abspos
        )
        args.hiddenGar = args.hiddenEncoder
    elif args.arMode == "no_ar":
        from .model import NoAr

        arNet = NoAr()
    else:
        from .model import CPCAR

        arNet = CPCAR(
            args.hiddenEncoder,
            args.hiddenGar,
            args.samplingType == "sequential",
            args.nLevelsGRU,
            mode=args.arMode,
            reverse=args.cpc_mode == "reverse",
        )
    return arNet


def loadModel(pathCheckpoints, loadStateDict=True):
    models = []
    hiddenGar, hiddenEncoder = 0, 0
    for path in pathCheckpoints:
        print(f"Loading checkpoint {path}")
        _, _, locArgs = getCheckpointData(os.path.dirname(path))

        doLoad = locArgs.load is not None and (
            len(locArgs.load) > 1
            or os.path.dirname(locArgs.load[0]) != os.path.dirname(path)
        )

        if doLoad:
            m_, hg, he = loadModel(locArgs.load, loadStateDict=False)
            hiddenGar += hg
            hiddenEncoder += he
        else:
            encoderNet = getEncoder(locArgs)

            arNet = getAR(locArgs)
            m_ = CPCModel(encoderNet, arNet)

        if loadStateDict:
            print(f"Loading the state dict at {path}")
            state_dict = torch.load(path, "cpu")
            m_.load_state_dict(state_dict["gEncoder"], strict=False)
        if not doLoad:
            hiddenGar += locArgs.hiddenGar
            hiddenEncoder += locArgs.hiddenEncoder

        models.append(m_)

    if len(models) == 1:
        return models[0], hiddenGar, hiddenEncoder

    return ConcatenatedModel(models), hiddenGar, hiddenEncoder


def get_module(i_module):
    if isinstance(i_module, torch.nn.DataParallel):
        return get_module(i_module.module)
    if isinstance(i_module, FeatureModule):
        return get_module(i_module.module)
    return i_module


def save_checkpoint(
    model_state, criterion_state, optimizer_state, best_state, path_checkpoint
):

    state_dict = {
        "gEncoder": model_state,
        "cpcCriterion": criterion_state,
        "optimizer": optimizer_state,
        "best": best_state,
    }

    torch.save(state_dict, path_checkpoint)


def toOneHot(inputVector, nItems):

    batchSize, seqSize = inputVector.size()
    out = torch.zeros(
        (batchSize, seqSize, nItems), device=inputVector.device, dtype=torch.long
    )
    out.scatter_(2, inputVector.view(batchSize, seqSize, 1), 1)
    return out


def seqNormalization(out):
    # out.size() = Batch x Seq x Channels
    mean = out.mean(dim=1, keepdim=True)
    var = out.var(dim=1, keepdim=True)
    return (out - mean) / torch.sqrt(var + 1e-08)


def buildFeature(featureMaker, seqPath, strict=False, maxSizeSeq=64000, seqNorm=False):
    r"""
    Apply the featureMaker to the given file.
    Arguments:
        - featureMaker (FeatureModule): model to apply
        - seqPath (string): path of the sequence to load
        - strict (bool): if True, always work with chunks of the size
                         maxSizeSeq
        - maxSizeSeq (int): maximal size of a chunk
        - seqNorm (bool): if True, normalize the output along the time
                          dimension to get chunks of mean zero and var 1
    Return:
        a torch vector of size 1 x Seq_size x Feature_dim
    """
    seq = torchaudio.load(seqPath)[0]
    sizeSeq = seq.size(1)
    start = 0
    out = []
    while start < sizeSeq:
        if strict and start + maxSizeSeq > sizeSeq:
            break
        end = min(sizeSeq, start + maxSizeSeq)
        subseq = (seq[:, start:end]).view(1, 1, -1).cuda(device=0)
        with torch.no_grad():
            features = featureMaker((subseq, None))
            if seqNorm:
                features = seqNormalization(features)
        out.append(features.detach().cpu())
        start += maxSizeSeq

    if strict and start < sizeSeq:
        subseq = (seq[:, -maxSizeSeq:]).view(1, 1, -1).cuda(device=0)
        with torch.no_grad():
            features = featureMaker((subseq, None))
            if seqNorm:
                features = seqNormalization(features)
        delta = (sizeSeq - start) // featureMaker.getDownsamplingFactor()
        out.append(features[:, -delta:].detach().cpu())

    out = torch.cat(out, dim=1)
    return out

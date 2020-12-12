import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mockingjay_default', help='The model variant')
parser.add_argument('--force_reload', action='store_true', help='Whether to re-download contents')
args = parser.parse_args()

REPO = 'andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning:hub'
model = torch.hub.load(REPO, args.model, force_reload=args.force_reload, use_cache=not args.force_reload)

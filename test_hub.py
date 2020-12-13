import argparse
import torch

import hubconf


parser = argparse.ArgumentParser()
upstreams = [attr for attr in dir(hubconf) if callable(getattr(hubconf, attr)) and attr[0] != '_']
parser.add_argument('--upstream', choices=upstreams, required=True)
parser.add_argument('--refresh', action='store_true', help='Whether to re-download upstream contents')
args = parser.parse_args()

REPO = 'andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning:benchmark'
model = torch.hub.load(REPO, args.upstream, force_reload=args.refresh, refresh=args.refresh)

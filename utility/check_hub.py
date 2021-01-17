import argparse
import torch

import hubconf


parser = argparse.ArgumentParser()
upstreams = [attr for attr in dir(hubconf) if callable(getattr(hubconf, attr)) and attr[0] != '_']
parser.add_argument('--mode', choices=['list', 'help', 'load'], required=True)
parser.add_argument('--upstream', choices=upstreams)
parser.add_argument('--ckpt', help='The PATH/URL/GOOGLE_DRIVE_ID of upstream checkpoint, not always needed')
parser.add_argument('--config', help='The PATH of upstream config, not always needed')
parser.add_argument('--refresh', action='store_true', help='Whether to re-download upstream contents')

args = parser.parse_args()

if args.mode == 'list':
    print(torch.hub.list('s3prl/s3prl', force_reload=args.refresh))

elif args.mode == 'help':
    print(torch.hub.help('s3prl/s3prl', args.upstream, force_reload=args.refresh))

elif args.mode == 'load':
    print(torch.hub.load(
        's3prl/s3prl', args.upstream, force_reload=args.refresh,
        ckpt=args.ckpt, config=args.config, refresh=args.refresh
    ))

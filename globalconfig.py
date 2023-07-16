import torch
import yaml
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs.yaml', help='Configuration File')
args = parser.parse_args()

with open(args.config, 'r') as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)

dpid = configs['training']['gpus']
device = torch.device(f'cuda:{dpid[0]}')
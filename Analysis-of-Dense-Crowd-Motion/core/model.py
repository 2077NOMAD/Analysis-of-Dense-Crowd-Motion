import torch
import torch.nn as nn
from models.raft import RAFT
from models.Localraft import ModifiedRAFT
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def generate_Raft_model(args):
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    return model

def generate_LocalRaft_model(args):
    model = nn.DataParallel(ModifiedRAFT(args), device_ids=args.gpus)

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    return model




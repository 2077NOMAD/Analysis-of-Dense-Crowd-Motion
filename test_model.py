import os
from opts import parse_opts
import torch
from torch.cuda.amp import GradScaler
import numpy as np


from Dataset.datasets import fetch_dataloader
from core.model import generate_Raft_model, generate_LocalRaft_model
from core.optimizer import fetch_optimizer
from core.loss import sequence_loss
from core.Logger import Logger
import evaluate


args = parse_opts()
batch_size = 2
channels = 3
height = 368
width = 768

# 随机生成两张图像作为输入
image1 = torch.randint(0, 256, (batch_size, channels, height, width), dtype=torch.uint8).cuda()
image2 = torch.randint(0, 256, (batch_size, channels, height, width), dtype=torch.uint8).cuda()

model = generate_Raft_model(args) 
model.cuda()
model.train()
model.module.freeze_bn()

model(image1, image2, iters=args.iters) 

# print(model)


    

import torch.nn as nn
from models.EmoEvent import EmoEvent
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def generate_Emo_model(opt):
    model=EmoEvent(
        opt,
        num_classes=opt.n_classes,
    )
    model = nn.DataParallel(model)
    model=model.cuda()
    return model, model.parameters()

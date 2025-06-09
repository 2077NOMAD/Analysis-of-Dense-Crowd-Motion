import torch.nn as nn
from models.EmoEvent import EmoEvent
from models.ReID import ReID
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def generate_Emo_model(opt):
    model = EmoEvent(
        opt,
        num_classes = opt.n_classes,
    )
    model = model.cuda()
    return model, model.parameters()


def generate_ReID_model(opt):
    model = ReID()
    model = model.cuda()
    return model, model.parameters()

# class GetModel:
#     def __init__(self):
#         self.model_name = None
    
#     def get_model(self, model_name):
#         pass

def get_model(opt):
    if opt.model == 'emo':
        return generate_Emo_model(opt)
    elif opt.model == 'deepsort':
        return generate_ReID_model(opt)
    else:
        raise ValueError(f"Unknown model type: {opt.model}")
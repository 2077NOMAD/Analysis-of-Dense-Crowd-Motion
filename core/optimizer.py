import torch.nn as nn
from torch import optim
from torch.optim import Adam, SGD


def get_optim(opt, parameters):
    if opt.optimizer == 'adam':
        optimizer = Adam(filter(lambda p: p.requires_grad, parameters),
                        lr=opt.learning_rate,
                        weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgd':
        pg = [p for p in parameters() if p.requires_grad]
        optimizer = SGD(pg, lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4)
    return optimizer

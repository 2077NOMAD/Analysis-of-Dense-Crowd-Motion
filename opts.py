import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'core': [
            dict(name='--name', default='raft-sintel', type=str),
            dict(name='--stage', default='sintel', type=str),
            dict(name='--restore_ckpt', default="/root/autodl-fs/Analysis-of-Dense-Crowd-Motion/checkpoints/raft-sintel.pth", type=str),
            dict(name='--small', action='store_true'),
            dict(name='--validation', default='sintel', type=str),

            dict(name='--lr', default=0.000125, type=float),
            dict(name='--num_steps', default=100000, type=int),
            dict(name='--batch_size', default=6, type=int),
            dict(name='--image_size', default=[368, 768], type=list),
            dict(name='--gpus', default=[0], type=list),
            dict(name='--mixed_precision', action='store_true'),

            dict(name='--iters', default=12, type=int),
            dict(name='--wdecay', default=0.00001, type=float),
            dict(name='--epsilon', default=1e-8, type=float),
            dict(name='--clip', default=1.0, type=float),
            dict(name='--dropout', default=0.0, type=float),
            dict(name='--gamma', default=0.85, type=float),
            dict(name='--add_noise', action='store_true'),
        ],
    }
    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)
    args = parser.parse_args()
    return args

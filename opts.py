import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'coefficients': [
            dict(name='--lambda_0',
                default='0.5',
                type=float,
                help='Penalty Coefficient that Controls the Penalty Extent in PCCE'),
          ],
          'paths': [
            dict(name='--resnet101_pretrained',
                 default='',
                 type=str,
                 help='Global path of pretrained 3d resnet101 model (.pth)'),
            dict(name='--root_path',
                 default="/root/autodl-fs/EMO",
                 type=str,
                 help='Global path of root directory'),
            dict(name="--video_path",
                 default="dataset/caer/fusion",
                 type=str,
                 help='Local path of videos', ),
            dict(name="--result_path",
                 default='results',
                 type=str,
                 help="Local path of result directory"),
        ],
        'core': [
            dict(name='--device',
                 default='cuda:0',
                 type=str,
                 help='cuda'),
            dict(name='--batch_size',
                 default=24,
                 type=int,
                 help='Batch Size'),
            dict(name='--n_classes',
                 default=7,
                 type=int,
                 help='Number of classes'),
            dict(name='--loss_func',
                 default='ce',
                 type=str,
                 help='ce'),
            dict(name='--learning_rate',
                 default=2e-4,
                 type=float,
                 help='Initial learning rate', ),
            dict(name='--weight_decay',
                 default=5e-4,
                 type=float,
                 help='Weight Decay'),
            dict(name='--optimizer',
                 default='adam',
                 type=str,
                 help='Optimizer'),
        ],
        'network': [
        ],
        'common': [
            dict(name='--class_to_idx',
                 action='store_true',
                 default={
                    "Anger": 0,
                    "Disgust": 1,
                    "Fear": 2,
                    "Happy": 3,
                    "Neutral": 4,
                    "Sad": 5,
                    "Surprise": 6
                },
                 help='class'),
            dict(name='--use_cuda',
                 action='store_true',
                 default=True,
                 help='only cuda supported!'
                 ),
            dict(name='--debug',
                 default=False,
                 action='store_true'),
            dict(name='--dl',
                 action='store_true',
                 default=True,
                 help='drop last'),
            dict(
                name='--n_threads',
                default=1,
                type=int,
                help='Number of threads for multi-thread loading',
            ),
            dict(
                name='--n_epochs',
                default=100,
                type=int,
                help='Number of total epochs to run',
            )
        ]
    }
    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)
    args = parser.parse_args()
    return args

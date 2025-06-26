import sys
import argparse
import os
from RAFT_demo import RAFT_demo
from DeepSORT_demo import DeepSORT_demo
from SocialVAE_demo import SocialVAE_demo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default="raft", help="restore checkpoint")

    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', default="/root/autodl-fs/WuhanMetro/clip2", help="dataset for evaluation")
    #parser.add_argument('--path', default="/root/autodl-fs/DEMO/1", help="dataset for evaluation")
    parser.add_argument('--output', default="/root/autodl-fs/DEMO/results/", help="output directory")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--traj_pre', default='./model/SocialVAE/wuhanmetro.txt')
    parser.add_argument('--output_form', default="flow", help="the output form of raft")
    args = parser.parse_args()

    if args.method == "raft":
        args.model = "/root/autodl-fs/RAFT/raft-sintel.pth"

        demo = RAFT_demo()
        demo.raft_demo(args)
    elif args.method == "deepsort":
        args.model = "/root/autodl-fs/DEMO/checkpoint/reid.t7"
        with DeepSORT_demo(args) as demo:
            demo.deepsort_demo()
    elif args.method == 'socialvae':
        args.model = 'checkpoint'
        demo = SocialVAE_demo(args)
        demo.socialvae_demo()
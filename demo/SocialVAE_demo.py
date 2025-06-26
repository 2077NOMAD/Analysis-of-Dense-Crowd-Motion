import os, sys, time
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import make_interp_spline

from model.SocialVAE.social_vae import SocialVAE
from model.SocialVAE.data import Dataloader
from model.SocialVAE.utils import get_rng_state, set_rng_state


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SocialVAE_demo:
    def __init__(self, args):
        spec = importlib.util.spec_from_file_location("config", 'model/SocialVAE/config.py')
        self.config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.config)

        self.args = args

        self.init_rng_state = get_rng_state(DEVICE)

        kwargs = dict(
                batch_first=False, frameskip=1,
                ob_horizon=self.config.OB_HORIZON, pred_horizon=self.config.PRED_HORIZON,
                device=DEVICE, seed=42)

        test = [args.traj_pre]
        if self.config.INCLUSIVE_GROUPS is not None:
            inclusive = [self.config.INCLUSIVE_GROUPS for _ in range(len(test))]
        else:
            inclusive = None
        test_dataset = Dataloader(
            test, **kwargs, inclusive_groups=inclusive,
            batch_size=self.config.BATCH_SIZE, shuffle=False
        )
        self.test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler
        )

    def test(self, model):
        self.out_path = self.args.output + "predictions.txt"
        model.eval()
        set_rng_state(self.init_rng_state, DEVICE)

        # 打开文件，准备写入
        with open(self.out_path, "w") as fout:
            with torch.no_grad():
                for x, y, neighbor, agent_ids, start_frames in self.test_data:
                    y_ = model(x, neighbor, n_predictions=self.config.PRED_SAMPLES)
                    if y_.dim() == 3:
                        y_ = y_.unsqueeze(0)
                    y_pred = y_[0]
                    L_pred, B, _ = y_pred.shape
                    for bi in range(B):
                        agent_id    = agent_ids[bi]
                        start_frame = start_frames[bi]
                        for t in range(L_pred):
                            frame_id = start_frame + t
                            x_pred, y_pred_coord = y_pred[t, bi].tolist()
                            fout.write(f"{frame_id} {agent_id} {x_pred:.3f} {y_pred_coord:.3f}\n")
    
    def visialize(self, gt_path, pred_path):
        # 读取数据
        origin_data = np.loadtxt(gt_path)
        predict_data = np.loadtxt(pred_path)


        H = np.array(
            [[0.02104651,   0,              0           ],
            [0,            -0.02386598,    13.74680446 ],
            [0,            0,              1           ]]
        )

        H = np.linalg.inv(H)

        # 选一段轨迹，投影到像素平面
        start_world = origin_data[origin_data[:,1]==72, 2:4][0:9]   # 取行人 id==3 的所有 (x,y)
        gt_world = origin_data[origin_data[:,1]==72, 2:4][9:21]   # 取行人 id==3 的所有 (x,y)
        predict_world = predict_data[predict_data[:,1]==72, 2:4][24:36]   # 取行人 id==3 的所有 (x,y)

        # 转齐次并映射
        def mapping(traj_world):
            ones      = np.ones((len(traj_world),1))
            homo_w    = np.hstack([traj_world, ones])
            homo_img  = (H @ homo_w.T).T
            traj_img  = homo_img[:,:2] / homo_img[:,2:3]
            return traj_img

        start_img = mapping(start_world)
        gt_img = mapping(gt_world)
        predict_img = mapping(predict_world)

        # 生成平滑的贝塞尔样条
        def spline(traj_img):
            t = np.arange(len(traj_img))  # 时间序列
            spl_x = make_interp_spline(t, traj_img[:,0])  # 对x坐标插值
            spl_y = make_interp_spline(t, traj_img[:,1])  # 对y坐标插值
            t_new = np.linspace(0, len(traj_img)-1, 100)  # 生成100个平滑点
            x_smooth = spl_x(t_new)  # 平滑后的x坐标
            y_smooth = spl_y(t_new)  # 平滑后的y坐标
            return x_smooth, y_smooth

        # 画到背景图上
        img = np.array(Image.open("model/SocialVAE/130.jpg"))
        plt.figure()
        plt.imshow(img)
        # plt.plot(traj_img[:,0], traj_img[:,1], '-o', color='red', label='Predicted')  # 原始轨迹
        # plt.plot(x_smooth, y_smooth, '-', color='blue', label='Smoothed')  # 平滑轨迹
        plt.plot(start_img[:,0], start_img[:,1], '-o', color='red', label='Start')
        plt.plot(gt_img[:,0], gt_img[:,1], '-o', color='green', label='Ground Truth')
        plt.plot(predict_img[:,0], predict_img[:,1], '-o', color='blue', label='Predicted')
        plt.legend()
        plt.axis('off')
        plt.savefig(self.args.output+"socialvae_demo.png")
            
    def socialvae_demo(self):
        model = SocialVAE(horizon=self.config.PRED_HORIZON, ob_radius=self.config.OB_RADIUS, hidden_dim=self.config.RNN_HIDDEN_DIM)
        model.to(DEVICE)
        ckpt_best = os.path.join(self.args.model, "socialvae")
        state_dict = torch.load(ckpt_best, map_location=DEVICE)
        model.load_state_dict(state_dict["model"])

        perform_test = self.test_data is not None
        if perform_test:
            self.test(model)
        print(self.args.traj_pre)
        print(self.out_path)
        self.visialize(self.args.traj_pre, self.out_path)


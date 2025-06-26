import os, sys, time
import importlib
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from social_vae import SocialVAE
from data import Dataloader
from utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no-fpc", action="store_true", default=False)
parser.add_argument("--fpc-finetune", action="store_true", default=False)
parser.add_argument("--save-pred", type=str, default=None,
                    help="Path to save predicted trajectories (NumPy .npz file or .txt file extension)")

if __name__ == "__main__":
    settings = parser.parse_args()
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.device = torch.device(settings.device)
    
    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    kwargs = dict(
            batch_first=False, frameskip=settings.frameskip,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=settings.device, seed=settings.seed)
    train_data, test_data = None, None
    if settings.test:
        print(settings.test)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None
        test_dataset = Dataloader(
            settings.test, **kwargs, inclusive_groups=inclusive,
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler
        )
        # def test(model, fpc):
        #     """
        #     测试函数：前向预测并保存 ETH/UCY 格式轨迹。
        #     返回平均 ADE 和 FDE。
        #     """
        #     model.eval()
        #     ADE_list, FDE_list = [], []
        #     all_preds, all_frames, all_pids = [], [], []
        #     set_rng_state(init_rng_state, settings.device)
        #     fpc = int(fpc) if fpc else 1
        #     with torch.no_grad():
        #         for x, y, neighbor, frames, pids in test_data:
        #             # y_: samples x T_pred x B x 2
        #             if config.PRED_SAMPLES > 0 and fpc > 1:
        #                 samples = []
        #                 for _ in range(fpc):
        #                     samples.append(model(x, neighbor, n_predictions=config.PRED_SAMPLES))
        #                 y_cat = torch.cat(samples, dim=0)
        #                 # select best by FPC
        #                 cand = [FPC(y_cat[..., i, :].cpu().numpy(), n_samples=config.PRED_SAMPLES)
        #                         for i in range(y_cat.size(-2))]
        #                 y_pred = torch.stack([y_cat[s, :, i] for i, s in enumerate(cand)], dim=2)
        #             else:
        #                 y_pred = model(x, neighbor, n_predictions=config.PRED_SAMPLES)
        #             # 误差计算
        #             ade, fde = ADE_FDE(y_pred, y)
        #             if config.PRED_SAMPLES > 0:
        #                 ade = torch.min(ade, dim=0)[0]
        #                 fde = torch.min(fde, dim=0)[0]
        #             ADE_list.append(ade)
        #             FDE_list.append(fde)
        #             # 收集预测、帧号、行人ID
        #             all_preds.append(y_pred[0].cpu().numpy())  # 取样本0: T_pred x B x 2
        #             all_frames.append(frames)                  # B x T_pred
        #             all_pids.append(pids)                      # B
        #     # 平均误差
        #     ADE = torch.cat(ADE_list).mean()
        #     FDE = torch.cat(FDE_list).mean()
        #     print(f"ADE: {ADE:.4f}, FDE: {FDE:.4f}")
        #     # 保存轨迹
        #     if settings.save_pred:
        #         # 拼接 batch
        #         preds = np.concatenate(all_preds, axis=1)      # T_pred x total_B x 2
        #         frames_arr = np.concatenate(all_frames, axis=0) # total_B x T_pred
        #         pids_arr = np.concatenate(all_pids, axis=0)     # total_B
        #         # 写入 ETH/UCY txt
        #         txt_path = settings.save_pred
        #         with open(txt_path, 'w') as f_txt:
        #             T, B, _ = preds.shape
        #             for i in range(B):
        #                 for t in range(T):
        #                     f_id = int(frames_arr[i, t])
        #                     pid = int(pids_arr[i])
        #                     x_, y_ = preds[t, i]
        #                     f_txt.write(f"{f_id} {pid} {x_:.4f} {y_:.4f}\n")
        #         print(f"Saved ETH/UCY format to {txt_path}")
        #     return ADE, FDE

        def test(model, fpc=1, out_path="predictions.txt"):
            model.eval()
            set_rng_state(init_rng_state, settings.device)

            # 打开文件，准备写入
            with open(out_path, "w") as fout:
                with torch.no_grad():
                    for x, y, neighbor, agent_ids, start_frames in test_data:
                        # x: [L_ob, B, 6], neighbor: ..., agent_ids: list of length B, start_frames: same
                        # 获取模型输出 y_: [n_samples, L_pred, B, 2]
                        y_ = model(x, neighbor, n_predictions=config.PRED_SAMPLES)
                        # 如果没有采样维度（n_samples=1），可以 squeeze
                        if y_.dim() == 3:
                            # [L_pred, B, 2]
                            y_ = y_.unsqueeze(0)
                        # 我们只取第一条预测轨迹（如果你有多条采样，可以在这里做均值/最优选择）
                        y_pred = y_[0]  # [L_pred, B, 2]

                        L_pred, B, _ = y_pred.shape
                        # 遍历 batch 中每个 agent
                        for bi in range(B):
                            agent_id    = agent_ids[bi]
                            start_frame = start_frames[bi]
                            for t in range(L_pred):
                                frame_id = start_frame + t
                                x_pred, y_pred_coord = y_pred[t, bi].tolist()
                                # 写成：frame_id agent_id x y
                                fout.write(f"{frame_id} {agent_id} {x_pred:.3f} {y_pred_coord:.3f}\n")


    if settings.train:
        print(settings.train)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.train))]
        else:
            inclusive = None
        train_dataset = Dataloader(
            settings.train, **kwargs, inclusive_groups=inclusive, 
            flip=True, rotate=True, scale=True,
            batch_size=config.BATCH_SIZE, shuffle=True, batches_per_epoch=config.EPOCH_BATCHES
        )
        train_data = torch.utils.data.DataLoader(train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_sampler=train_dataset.batch_sampler
        )
        batches = train_dataset.batches_per_epoch

    ###############################################################################
    #####                                                                    ######
    ##### load model                                                         ######
    #####                                                                    ######
    ###############################################################################
    model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
    model.to(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    start_epoch = 0
    if settings.ckpt:
        ckpt = os.path.join(settings.ckpt, "ckpt-last")
        ckpt_best = os.path.join(settings.ckpt, "ckpt-best")
        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best, map_location=settings.device)
            ade_best = state_dict["ade"]
            fde_best = state_dict["fde"]
            fpc_best = state_dict["fpc"] if "fpc" in state_dict else 1
        else:
            ade_best = 100000
            fde_best = 100000
            fpc_best = 1
        if train_data is None: # testing mode
            ckpt = ckpt_best
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            state_dict = torch.load(ckpt, map_location=settings.device)
            model.load_state_dict(state_dict["model"])
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
                rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
            start_epoch = state_dict["epoch"]
    end_epoch = start_epoch+1 if train_data is None or start_epoch >= config.EPOCHS else config.EPOCHS

    if settings.train and settings.ckpt:
        logger = SummaryWriter(log_dir=settings.ckpt)
    else:
        logger = None

    if train_data is not None:
        log_str = "\r\033[K {cur_batch:>"+str(len(str(batches)))+"}/"+str(batches)+" [{done}{remain}] -- time: {time}s - {comment}"    
        progress = 20/batches if batches > 20 else 1
        optimizer.zero_grad()

    for epoch in range(start_epoch+1, end_epoch+1):
        ###############################################################################
        #####                                                                    ######
        ##### train                                                              ######
        #####                                                                    ######
        ###############################################################################
        losses = None
        if train_data is not None and epoch <= config.EPOCHS:
            print("Epoch {}/{}".format(epoch, config.EPOCHS))
            tic = time.time()
            set_rng_state(rng_state, settings.device)
            losses = {}
            model.train()
            sys.stdout.write(log_str.format(
                cur_batch=0, done="", remain="."*int(batches*progress),
                time=round(time.time()-tic), comment=""))
            for batch, item in enumerate(train_data):
                res = model(*item)
                loss = model.loss(*res)
                loss["loss"].backward()
                optimizer.step()
                optimizer.zero_grad()
                for k, v in loss.items():
                    if k not in losses: 
                        losses[k] = v.item()
                    else:
                        losses[k] = (losses[k]*batch+v.item())/(batch+1)
                sys.stdout.write(log_str.format(
                    cur_batch=batch+1, done="="*int((batch+1)*progress),
                    remain="."*(int(batches*progress)-int((batch+1)*progress)),
                    time=round(time.time()-tic),
                    comment=" - ".join(["{}: {:.4f}".format(k, v) for k, v in losses.items()])
                ))
            rng_state = get_rng_state(settings.device)
            print()

        ###############################################################################
        #####                                                                    ######
        ##### test                                                               ######
        #####                                                                    ######
        ###############################################################################
        # ade, fde = 10000, 10000
        perform_test = (train_data is None or epoch >= config.TEST_SINCE) and test_data is not None
        if perform_test:
        #     if not settings.no_fpc and not settings.fpc_finetune and losses is None and fpc_best > 1:
        #         fpc = fpc_best
        #     else:
        #         fpc = 1
        #     ade, fde = test(model, fpc)
            test(model, fpc=1)

        ###############################################################################
        #####                                                                    ######
        ##### log                                                                ######
        #####                                                                    ######
        ###############################################################################
    #     if losses is not None and settings.ckpt:
    #         if logger is not None:
    #             for k, v in losses.items():
    #                 logger.add_scalar("train/{}".format(k), v, epoch)
    #             if perform_test:
    #                 logger.add_scalar("eval/ADE", ade, epoch)
    #                 logger.add_scalar("eval/FDE", fde, epoch)
    #         state = dict(
    #             model=model.state_dict(),
    #             optimizer=optimizer.state_dict(),
    #             ade=ade, fde=fde, epoch=epoch, rng_state=rng_state
    #         )
    #         torch.save(state, ckpt)
    #         if ade < ade_best:
    #             ade_best = ade
    #             fde_best = fde
    #             state = dict(
    #                 model=state["model"],
    #                 ade=ade, fde=fde, epoch=epoch
    #             )
    #             torch.save(state, ckpt_best)

    # if settings.fpc_finetune or losses is not None:
    #     # FPC finetune if it is specified or after training
    #     precision = 2
    #     trunc = lambda v: np.trunc(v*10**precision)/10**precision
    #     ade_, fde_, fpc_ = [], [], []
    #     for fpc in config.FPC_SEARCH_RANGE:
    #         ade, fde = test(model, fpc)
    #         ade_.append(trunc(ade.item()))
    #         fde_.append(trunc(fde.item()))
    #         fpc_.append(fpc)
    #     i = np.argmin(np.add(ade_, fde_))
    #     ade, fde, fpc = ade_[i], fde_[i], fpc_[i]
    #     if settings.ckpt:
    #         ckpt_best = os.path.join(settings.ckpt, "ckpt-best")
    #         if os.path.exists(ckpt_best):
    #             state_dict = torch.load(ckpt_best, map_location=settings.device)
    #             state_dict["ade_fpc"] = ade
    #             state_dict["fde_fpc"] = fde
    #             state_dict["fpc"] = fpc
    #             torch.save(state_dict, ckpt_best)
    #     print(" ADE: {:.2f}; FDE: {:.2f} ({})".format(
    #         ade, fde, "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC", 
    #     ))
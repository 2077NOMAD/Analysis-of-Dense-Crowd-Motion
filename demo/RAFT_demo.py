import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from model.RAFT.raft import RAFT
from model.RAFT.utils import flow_viz
from model.RAFT.utils.utils import InputPadder


DEVICE = 'cuda'

class RAFT_demo:
    def load_image(self, imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def viz1(self, img, flo, output_dir, index):
        img = img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
        flo = flo[0].permute(1,2,0).cpu().numpy()

        # 创建原始图像的副本用于绘制
        img_with_arrows = img.copy()

        # 光流缩放因子
        scale_factor = 0.1

        # 获取光流幅度和方向
        magnitude = np.sqrt(flo[...,0]**2 + flo[...,1]**2)
        angle = np.arctan2(flo[...,1], flo[...,0])

        # 筛选显著运动的点
        min_magnitude = magnitude.mean() * 0.5
        valid_indices = np.where(magnitude > min_magnitude)
        y_coords, x_coords = valid_indices[0], valid_indices[1]

        # 如果没有显著运动的点，直接返回图像
        if len(y_coords) == 0:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"result_{index:04d}.png")
            cv2.imwrite(output_path, img_with_arrows[:, :, [2, 1, 0]])
            return img_with_arrows

        # 创建空间网格进行均匀采样
        grid_size = 20  # 网格大小（像素）
        h, w = img.shape[:2]
        grid_x = np.arange(0, w, grid_size)
        grid_y = np.arange(0, h, grid_size)

        # 存储每个网格中的候选点
        grid_points = {}
        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]
            grid_i = int(x / grid_size)
            grid_j = int(y / grid_size)

            # 只考虑网格内幅度最大的点
            if (grid_i, grid_j) not in grid_points:
                grid_points[(grid_i, grid_j)] = (x, y, magnitude[y, x])
            else:
                _, _, current_mag = grid_points[(grid_i, grid_j)]
                if magnitude[y, x] > current_mag:
                    grid_points[(grid_i, grid_j)] = (x, y, magnitude[y, x])

        # 随机选择部分网格点（最多300个）
        selected_points = list(grid_points.values())
        if len(selected_points) > 400:
            indices = np.random.choice(len(selected_points), 300, replace=False)
            selected_points = [selected_points[i] for i in indices]

        # 绘制选中的点
        for point in selected_points:
            x, y, mag = point
            ang = angle[y, x]

            # 计算箭头终点
            fx = mag * np.cos(ang) * scale_factor
            fy = mag * np.sin(ang) * scale_factor
            end_x = int(x + fx)
            end_y = int(y + fy)

            # 根据方向计算颜色 (HSV空间)
            hue = int(((ang + np.pi) / (2 * np.pi)) * 180)  # 0-180
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(map(int, color))

            # 绘制圆点标记起始位置
            cv2.circle(img_with_arrows, (x, y), 3, color, -1)  # -1表示实心圆

            # 绘制箭头
            cv2.arrowedLine(
                img_with_arrows, 
                (x, y), 
                (end_x, end_y), 
                color, 
                2, 
                tipLength=0.3
            )

        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{index:04d}.png")
        cv2.imwrite(output_path, img_with_arrows[:, :, [2, 1, 0]])

        return img_with_arrows

    def viz2(self, img, flo, output_dir, index):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()
        flo = flow_viz.flow_to_image(flo)
        # img_flo = np.concatenate([img, flo], axis=0)
        img_flo = flo
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{index:04d}.png")
        cv2.imwrite(output_path, img_flo[:, :, [2, 1, 0]])
        return img_flo


    def save_flow(self, flo, flow_dir, index):
        """保存光流数据到.flo文件"""
        os.makedirs(flow_dir, exist_ok=True)
        output_path = os.path.join(flow_dir, f"{index:03d}.flo")
        
        # 将光流数据从tensor转换为numpy数组
        flo_np = flo[0].permute(1, 2, 0).cpu().numpy()
        
        # 创建光流文件头部 (TAG_FLOAT)
        tag = np.array([202021.25], dtype=np.float32)
        height, width = flo_np.shape[:2]
        
        # 写入文件
        with open(output_path, 'wb') as f:
            f.write(tag.tobytes())
            f.write(np.int32(width).tobytes())
            f.write(np.int32(height).tobytes())
            f.write(flo_np.tobytes())
            
        print(f"已保存光流文件: {output_path}")

    def create_video_from_images(self, image_dir, output_video, fps=3):
        """从图像序列创建视频"""
        # 获取所有图像文件并排序
        image_files = sorted(glob.glob(os.path.join(image_dir, "result_*.png")))
        if not image_files:
            print(f"警告: 在 {image_dir} 中未找到结果图像")
            return False
        
        # 读取第一张图像获取尺寸
        first_image = cv2.imread(image_files[0])
        height, width, layers = first_image.shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于MP4格式
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # 逐帧写入视频
        for image_file in image_files:
            img = cv2.imread(image_file)
            if img is not None:
                video_writer.write(img)
        
        # 释放资源
        video_writer.release()
        print(f"视频已保存至: {output_video}")
        return True

    def raft_demo(self, args):
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))
        model = model.module
        model.to(DEVICE)
        model.eval()

        output_dir = os.path.join(args.output, "raft")
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建光流保存目录
        flow_dir = os.path.join(args.output, "flow")
        os.makedirs(flow_dir, exist_ok=True)
        
        raft_output = os.path.join(args.output, "raft.mp4")
        os.makedirs(os.path.dirname(raft_output), exist_ok=True)

        with torch.no_grad():
            images = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))
            
            images = sorted(images)
            for index, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
                image1 = self.load_image(imfile1)
                image2 = self.load_image(imfile2)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                # self.viz1(image1, flow_up, output_dir, index)
                # self.viz2(image1, flow_up, output_dir, index)
                if args.output_form == "arrow":
                    self.viz1(image1, flow_up, output_dir, index)
                elif args.output_form == "flow":
                    self.viz2(image1, flow_up, output_dir, index)
                else:
                    print('Form Erroe!')
                    break
                
                # 保存光流文件
                self.save_flow(flow_up, flow_dir, index)
                
        self.create_video_from_images(output_dir, raft_output)    
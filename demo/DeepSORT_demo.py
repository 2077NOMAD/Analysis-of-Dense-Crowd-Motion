import os
import sys
import time
import json
import glob
import warnings
import argparse
import cv2
import torch
import numpy as np
from PIL import Image

from model.DeepSORT.detector import build_detection
from model.DeepSORT import build_tracker
from model.DeepSORT.utils.draw import draw_boxes
from model.DeepSORT.utils.log import get_logger
from model.DeepSORT.utils.parser import get_config
from model.DeepSORT.utils.io import write_results


DEVICE = 'cuda'


def get_cfg():
    cfg = get_config()
    cfg.USE_SEGMENT = True
    cfg.USE_MMDET = False
    cfg.merge_from_file("model/DeepSORT/configs/yolov5m.yaml")
    cfg.merge_from_file("model/DeepSORT/configs/deep_sort.yaml")
    cfg.USE_FASTREID = False
    return cfg

class DeepSORT_demo:
    def __init__(self, opt):
        self.cfg = get_cfg()
        self.opt = opt
        self.logger = get_logger("root")

        use_cuda = DEVICE != "cpu" and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Run in cpu mode, which may be very slow!", UserWarning)
        
        self.video = cv2.VideoCapture()

        self.detector = build_detection(self.cfg)
        self.deepsort = build_tracker(self.cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.opt.output:
            self.output_dir = os.path.join(self.opt.output, "deepsort")
            os.makedirs(self.output_dir, exist_ok=True)

            self.deepsort_output = os.path.join(self.opt.output, "deepsort.mp4")
            os.makedirs(os.path.dirname(self.deepsort_output), exist_ok=True)

        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_type, exc_traceback)

    def load_image(self,imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        return img

    def create_video_from_images(self, image_dir, output_video, fps=5):
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

    def deepsort_demo(self, frame_interval=1):

        with open('model/DeepSORT/coco_classes.json', 'r') as f:
            idx_to_class = json.load(f)

        images = glob.glob(os.path.join(self.opt.path, '*.png')) + \
                glob.glob(os.path.join(self.opt.path, '*.jpg'))
        images = sorted(images)

        for index, ori_im in enumerate(images):
            if index % frame_interval:
                continue

            ori_im = self.load_image(ori_im)

            img = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            img = ori_im

            segment = False
            if segment:
                bbox_xywh, cls_conf, cls_ids, seg_masks = self.detector(img)
            else:
                bbox_xywh, cls_conf, cls_ids = self.detector(img)
            
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            bbox_xywh[:, 2:] *= 1.2
            cls_conf = cls_conf[mask]
            cls_ids = cls_ids[mask]

            if segment:
                seg_masks = seg_masks[mask]
                outputs, mask_outputs = self.deepsort.update(bbox_xywh, cls_conf, 
                                                             cls_ids, img, seg_masks)
            else:
                outputs, _ = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, img)

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                cls = outputs[:, -2]
                names = [idx_to_class[str(label)] for label in cls]

                ori_im = draw_boxes(ori_im, bbox_xyxy, names, identities, 
                                    None if not segment else mask_outputs)
                ori_im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                output_path = os.path.join(self.output_dir, f"result_{index:04d}.png")
                cv2.imwrite(output_path, ori_im)
        self.create_video_from_images(self.output_dir, self.deepsort_output)
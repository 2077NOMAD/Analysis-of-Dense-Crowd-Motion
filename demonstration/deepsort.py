import os
import cv2
import time
import torch
import warnings
import json

# from .deep_sort.detector import build_detection
from .deep_sort.detector import build_detection
from .deep_sort import build_tracker
from .deep_sort.utils.log import get_logger
from .deep_sort.utils.draw import draw_boxes
from .deep_sort.utils.io import write_results


class VideoTracker:
    def __init__(self, cfg, opt, video_path):
        self.cfg = cfg
        self.opt = opt
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = opt.device != "cpu" and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Run in cpu mode, which may be very slow!", UserWarning)
        
        self.video = cv2.VideoCapture()

        self.detector = build_detection(cfg, use_cuda=use_cuda, segment=self.opt.segment)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        assert os.path.isfile(self.video_path), "Path Error"
        self.video.open(self.video_path)
        self.im_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.video.isOpened()

        if self.opt.save_path:
            os.makedirs(self.opt.save_path, exist_ok=True)

            output_avi = "avi/" + self.opt.output_name + ".avi"
            output_txt = "txt/" + self.opt.output_name + ".txt"

            self.save_video_path = os.path.join(self.opt.save_path, output_avi)
            self.save_results_path = os.path.join(self.opt.save_path, output_txt)

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, 
                                          (self.im_width, self.im_height))
            
            self.logger.info("Save results to {}".format(self.opt.save_path))

        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_type, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0

        with open('demonstration/deep_sort/coco_classes.json', 'r') as f:
            idx_to_class = json.load(f)
        
        while self.video.grab():
            idx_frame += 1
            if idx_frame % self.opt.frame_interval:
                continue
        
            start = time.time()
            _, ori_im = self.video.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            if self.opt.segment:
                bbox_xywh, cls_conf, cls_ids, seg_masks = self.detector(im)
            else:
                bbox_xywh, cls_conf, cls_ids = self.detector(im)
            
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            bbox_xywh[:, 2:] *= 1.2
            cls_conf = cls_conf[mask]
            cls_ids = cls_ids[mask]

            if self.opt.segment:
                seg_masks = seg_masks[mask]
                outputs, mask_outputs = self.deepsort.update(bbox_xywh, cls_conf, 
                                                             cls_ids, im, seg_masks)
            else:
                outputs, _ = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)

            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                cls = outputs[:, -2]
                names = [idx_to_class[str(label)] for label in cls]

                ori_im = draw_boxes(ori_im, bbox_xyxy, names, identities, 
                                    None if not self.opt.segment else mask_outputs)
                
                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append([self.deepsort._xyxy_to_tlwh(bb_xyxy)])
                
                results.append((idx_frame-1, bbox_tlwh, identities, cls))
            
            end = time.time()

            if self.opt.save_path:
                self.writer.write(ori_im)
            
            write_results(self.save_results_path, results, 'mot')

            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))

from . import deepsort


def get_demo(opt, video_path):
    if opt.model == 'deepsort':

        from .deep_sort.utils.parser import get_config

        cfg = get_config()

        cfg.USE_SEGMENT = True
        cfg.USE_MMDET = False
        cfg.merge_from_file("./demonstration/deep_sort/configs/yolov5m.yaml")
        cfg.merge_from_file("./demonstration/deep_sort/configs/deep_sort.yaml")
        cfg.USE_FASTREID = False

        demo = deepsort.VideoTracker(cfg, opt, video_path)
        return demo
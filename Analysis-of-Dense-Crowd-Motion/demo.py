from demonstration.get_demo import get_demo
from opts import parse_opts


def demo(opt):
    demo_model = get_demo(opt, video_path=opt.video_path)
    # print("Starting demo...")
    demo_model.run()


if __name__ == "__main__":
    opt = parse_opts()
    demo(opt)
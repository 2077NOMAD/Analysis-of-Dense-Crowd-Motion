import os

def local2global_path(opt):
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        opt.tboard_path = os.path.join(opt.result_path, "tensorboard")
        opt.ckpt_path = os.path.join(opt.result_path, "checkpoints")
        opt.log_path = os.path.join(opt.result_path, "logs")
        opt.model_path = os.path.join(opt.result_path, "models")
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
        if not os.path.exists(opt.ckpt_path):
            os.mkdir(opt.ckpt_path)
        if not os.path.exists(opt.tboard_path):
            os.mkdir(opt.tboard_path)
        if not os.path.exists(opt.model_path):
            os.mkdir(opt.model_path)
    else:
        raise Exception



def run_emo(opt, inputs, model, criterion,labels):
    vit_embeds = inputs['Video']
    llm_embeds = inputs['LLM']
    output = model(vit_embeds=vit_embeds, llm_embeds=llm_embeds)
    loss = criterion(output, labels)
    return output,loss


def run_deepsort(opt, inputs, model):
    pass


def run_model(opt, inputs, model, criterion=None, labels=None):
    if opt.model == 'emo':
        return run_emo(opt, inputs, model, criterion, labels)
    elif opt.model == 'deepsort':
        return run_deepsort(opt, inputs, model)
    else:
        raise ValueError(f"Unknown model type: {opt.model}")
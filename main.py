from tensorboardX import SummaryWriter
import logging
import datetime
from train import train_one_epoch
from evaluate import evaluate
from core.dataloader import get_training_set, get_validation_set, get_test_set, get_data_loader
from core.model import generate_Emo_model
from opts import parse_opts
from core.optimizer import get_optim
from core.loss import get_loss
from torch.cuda import device_count
from core.utils import local2global_path
import os


def main():
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    local2global_path(opt)
    model,parameters = generate_Emo_model(opt) #
    criterion = get_loss(opt) #
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)
    writer = SummaryWriter(logdir=os.path.join(opt.tboard_path, timestamp))
    spatial_transform = None
    temporal_transform = None
    target_transform = None

    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform) #
    train_loader = get_data_loader(opt, training_data, shuffle=True)
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)

    log_file = os.path.join(opt.log_path, "/train_{timestamp}.log")
    logging.basicConfig(
        filename=log_file, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        force=True  
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    
    best_acc = 0.0
    writer = SummaryWriter(log_dir=os.path.join(opt.tboard_path, timestamp))
    for i in range(1, opt.n_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            epoch=i,
            data_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            opt=opt,
            writer=writer
        )
        val_loss, val_acc = evaluate(
            epoch=i,
            data_loader=val_loader,
            model=model,
            criterion=criterion,
            opt=opt,
            writer=writer
        )
        logging.info(f"Epoch {i}/{opt.n_epochs} :")
        logging.info(f"Train_loss: {train_loss:.4f}----Val_loss: {val_loss:.4f}")
        logging.info(f"Train_acc: {train_acc*100:.2f}%----Val_acc: {val_acc*100:.2f}%")
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(), opt.model_path + "/best_model.pth")
            # logging.info(f"Best model saved, val_acc: {val_acc*100:.2f}%")
        logging.info(f"best_acc: {best_acc*100:.2f}%\n\n")
    writer.close()


# 修改训练参数部分
if __name__ == '__main__':
    main()

from tqdm import tqdm
from core.utils import run_model

def train_one_epoch(epoch, data_loader, model, criterion, optimizer, opt, writer):  # 添加args参数
    print("# ---------------------------------------------------------------------- #")
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    train_bar = tqdm(data_loader, desc=f"Epoch {epoch}/{opt.n_epochs}")
    for i, (inputs, labels) in enumerate(train_bar): 
        labels = labels.to(opt.device)
        output,loss = run_model(opt, inputs, model, criterion,labels)
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        total_loss += loss.item() 
        _, predicted = output.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        train_bar.set_postfix({
            'batch_loss': f"{loss.item():.4f}", 
            'avg_loss': f"{total_loss/(train_bar.n+1):.4f}" 
        })
    avg_loss = total_loss / len(data_loader)
    acc = correct / total
    writer.add_scalar('train/epoch/loss', avg_loss, epoch)
    writer.add_scalar('train/epoch/acc', acc, epoch)
    return avg_loss, acc



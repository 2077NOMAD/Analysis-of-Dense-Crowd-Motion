from tqdm import tqdm
from core.utils import run_model

def evaluate(epoch, data_loader, model, criterion, opt, writer):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        val_bar = tqdm(data_loader, desc=f"Validating Epoch {epoch}")
        for batch_idx, (inputs, labels) in enumerate(val_bar):
            labels = labels.to(opt.device)
            output,loss = run_model(opt, inputs, model, criterion,labels)
            total_loss += loss.item()
            _, predicted = output.max(1)
            correct_mask = predicted.eq(labels)
            correct += correct_mask.sum().item()
            total += labels.size(0)
            val_bar.set_postfix({
                'val_loss': f"{loss.item():.4f}",
                'val_acc': f"{correct/total*100:.2f}%"
            })
    avg_loss = total_loss / len(data_loader)
    acc = correct / total
    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/acc', acc, epoch)
    return avg_loss, acc
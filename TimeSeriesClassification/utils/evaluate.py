import torch


def evaluate(val_loader, model, classifier, loss, device):
    val_loss = 0
    val_accu = 0

    sum_len = 0
    for data, target in val_loader:
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        with torch.no_grad():
            val_pred = model(data)
            val_pred = classifier(val_pred)
            val_loss += loss(val_pred, target).item()

            val_accu += torch.sum(torch.argmax(val_pred.data, axis=1) == target)
            sum_len += len(target)

    return val_loss / sum_len, val_accu / sum_len

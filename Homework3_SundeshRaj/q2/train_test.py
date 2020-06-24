## Do NOT modify the code in this file
import torch
from datetime import datetime
import numpy as np


class Accuracy:
    def __init__(self, eps=1e-7):
        self._correct = 0
        self._total = 0
        self._eps = eps

    def update(self, preds, targets):
        if preds.size(1) == 1:
            preds = torch.sigmoid(preds)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
        else:
            preds = torch.argmax(preds, dim=1)
        self._correct += (preds == targets).sum().item()
        self._total += len(targets)

    def get(self):
        return self._correct / (self._total + self._eps)

    def reset(self):
        self._correct = 0
        self._total = 0


## train the model
def train(dataloader, model, criterion, optimizer, scheduler, args, epoch):
    model.train()

    train_loss_history = []
    train_acc_metric = Accuracy()
    start_time = datetime.now().timestamp()
    for batch_idx, (image, label) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            train_loss_history.append(loss.item())
            train_acc_metric.update(pred, label)

        if batch_idx % args.print_interval == 0:
            speed = args.batch_size * args.print_interval / (
                datetime.now().timestamp() - start_time)
            start_time = datetime.now().timestamp()
            
    return np.mean(train_loss_history), train_acc_metric.get()


def test(dataloader, model, criterion, epoch):
    print(f"{datetime.now().ctime()} - Start Testing Neural Model...")
    model.eval()

    loss_history = []
    acc_metric = Accuracy()

    with torch.no_grad():
        for image, label in dataloader:
            pred = model(image)
            loss = criterion(pred, label)

            loss_history.append(loss.item())
            acc_metric.update(pred, label)
    save_model(model, acc_metric.get())
    return np.mean(loss_history), acc_metric.get()


def lr_train(dataloader, model, criterion, optimizer, scheduler, args, epoch):
    model.train()

    loss_history = []
    acc_metric = Accuracy()
    start_time = datetime.now().timestamp()
    for batch_idx, (image, label) in enumerate(dataloader, 1):
        label = label.unsqueeze(1).to(torch.float32)
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            loss_history.append(loss.item())
            acc_metric.update(pred, label)

        if batch_idx % args.print_interval == 0:
            speed = args.batch_size * args.print_interval / (
                datetime.now().timestamp() - start_time)
            loss_history = []
            acc_metric.reset()
            start_time = datetime.now().timestamp()


def lr_test(dataloader, model, criterion, epoch):
    print(f"{datetime.now().ctime()} - Start Testing LR Model...")
    model.eval()

    loss_history = []
    acc_metric = Accuracy()

    with torch.no_grad():
        for image, label in dataloader:
            label = label.unsqueeze(1).to(torch.float32)
            pred = model(image)
            loss = criterion(pred, label)

            loss_history.append(loss.item())
            acc_metric.update(pred, label)
    save_model(model, acc_metric.get())
    return np.mean(loss_history), acc_metric.get()


def save_model(model, metric):
    cur_time = datetime.now()
    datetime_prefix = f"{cur_time.year}{cur_time.month}{cur_time.day}{cur_time.hour}{cur_time.minute}{cur_time.second}"
    model_filename = f"{datetime_prefix}_{metric:5f}.pth"
    torch.save(model.state_dict(), f"model_weight_image/{model_filename}")
    print(f"{datetime.now().ctime()} - Model Saved")

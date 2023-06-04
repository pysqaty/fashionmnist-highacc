import torch
import time
import os
from const import SEED, NEPOCHS, PATIENCE, DO_TRAINING
from model import Model, ResNet18Model
from dataset import Dataset, FashionMnist
from typing import Tuple


def run(model: ResNet18Model, loader: torch.utils.data.DataLoader, train: bool = True) -> Tuple[float, float]:
    """One iteration of training loop / Model evaluation"""
    running_loss = 0
    running_accuracy = 0

    # Set mode
    if train:
        model.net.train()
    else:
        model.net.eval()

    for i, (X, y) in enumerate(loader):
        # Pass to gpu or cpu
        X, y = X.to(device), y.to(device)

        # Zero the gradient
        model.optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            output = model.net(X)
            _, pred = torch.max(output, 1)
            loss = model.criterion(output, y)

        # If on train backpropagate
        if train:
            loss.backward()
            model.optimizer.step()

        # Calculate stats
        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())
    return running_loss / len(loader), running_accuracy.double() / len(loader.dataset)


if __name__ == "__main__":
    """Initialization"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Calculating on {}".format(device))

    torch.manual_seed(SEED)
    if device == "cuda:0":
        torch.cuda.manual_seed(SEED)

    if not os.path.exists("models/"):
        os.mkdir("models/")

    """ Data preparation """
    dataset: Dataset = FashionMnist(path="./data/")

    """ Model preparation """
    model: Model = ResNet18Model(device)

    best_acc = 0.0

    if not DO_TRAINING:
        start = time.time()
        train_loss, train_acc = run(model, dataset.get_train_loader(), False)
        val_loss, val_acc = run(model, dataset.val_loader, False)
        end = time.time()
        stats = """Epoch: {}\t train loss: {:.3f}, train acc: {:.4f}\t
                    val loss: {:.3f}, val acc: {:.4f}\t
                    time: {:.1f}s""".format(
            0, train_loss, train_acc, val_loss, val_acc, end - start
        )
        print(stats)
        exit(0)

    """ Training loop """
    for e in range(NEPOCHS):
        start = time.time()
        train_loss, train_acc = run(model, dataset.get_train_loader())
        val_loss, val_acc = run(model, dataset.get_val_loader(), False)
        end = time.time()
        stats = """Epoch: {}\t train loss: {:.3f}, train acc: {:.4f}\t
                val loss: {:.3f}, val acc: {:.4f}\t
                time: {:.1f}s""".format(
            e + 1, train_loss, train_acc, val_loss, val_acc, end - start
        )
        print(stats)
        if val_acc > best_acc:
            best_acc = val_acc
            current_patience = PATIENCE
            model.save("models/model-{}.pth.tar".format(best_acc))
        else:
            current_patience -= 1
            if current_patience == 0:
                print("Run out of patience!")
                break

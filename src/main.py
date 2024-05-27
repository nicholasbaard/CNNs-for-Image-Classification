import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse

from lenet import LeNet5
from get_mnist import get_mnist
from train_model import train, test


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--gamma', type=float, default=0.7, help='Gamma value for the loss function')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')

    args = parser.parse_args()


    best_loss = float('inf')
    epochs_no_improve = 0

    # set flags / seeds
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get data
    train_dataloader, test_dataloader = get_mnist(batch_size=args.batch_size)

    # initialize NN
    model = LeNet5().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    critrion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    writer = SummaryWriter()

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", position=0, leave=True):
        train(model, device, train_dataloader, optimizer, epoch, critrion, writer)
        test_loss = test(model, device, test_dataloader, epoch, critrion, writer)
        scheduler.step()

        # Early stopping logic
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "../models/mnist_cnn_best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f'Early stopping after {epoch} epochs.')
            break
    
    # Optionally save the final model
    if args.save_model:
        torch.save(model.state_dict(), "../models/mnist_cnn_final.pt")

if __name__ == '__main__':
    main()


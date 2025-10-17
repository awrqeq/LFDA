# =================================================================================================
# train_clean_model.py (Import修复 & 最终版)
# =================================================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import os

# [修复] 从 data.cifar10 导入正确的 get_dataloader 函数
from data.cifar10 import get_dataloader
from core.models.resnet import ResNet18


def train_clean(cfg):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories for saving models
    os.makedirs(cfg['train_clean']['save_dir'], exist_ok=True)

    # Load CIFAR-10 dataset using the correct function
    train_loader = get_dataloader(
        batch_size=cfg['train_clean']['batch_size'],
        train=True,
        path=cfg['dataset']['root_dir'],
        num_workers=cfg['dataset']['num_workers']
    )
    test_loader = get_dataloader(
        batch_size=cfg['train_clean']['batch_size'],
        train=False,
        path=cfg['dataset']['root_dir'],
        num_workers=cfg['dataset']['num_workers']
    )

    # Build model
    model = ResNet18(num_classes=cfg['dataset']['num_classes']).to(device)

    # Setup optimizer and loss function
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['train_clean']['optimizer']['lr'],
        momentum=cfg['train_clean']['optimizer']['momentum'],
        weight_decay=cfg['train_clean']['optimizer']['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train_clean']['epochs'])
    criterion = nn.CrossEntropyLoss()

    print("--- Starting Clean Model Training ---")
    for epoch in range(cfg['train_clean']['epochs']):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['train_clean']['epochs']}")

        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{(running_loss / (progress_bar.n + 1)):.4f}'})

        scheduler.step()

        # Evaluate on test set every few epochs
        if (epoch + 1) % cfg['train_clean']['eval_every_epochs'] == 0 or (epoch + 1) == cfg['train_clean']['epochs']:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    outputs = model(x_test)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_test.size(0)
                    correct += (predicted == y_test).sum().item()
            accuracy = 100 * correct / total
            print(f"\nEpoch {epoch + 1}: Test Accuracy = {accuracy:.2f}%")

    # Save the final model
    save_path = os.path.join(cfg['train_clean']['save_dir'], cfg['train_clean']['save_name'])
    torch.save(model.state_dict(), save_path)
    print(f"Final clean model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a clean model on CIFAR-10")
    parser.add_argument('--config', type=str, default='configs/cifar10_clean.yml',
                        help='Path to the config file for clean training')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    train_clean(cfg)
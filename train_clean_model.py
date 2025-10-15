import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from core.models.resnet import ResNet18
from data.cifar10 import get_dataloader


def main(config_path):
    # Load config, but we only need a few parameters from it
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])

    # --- Use victim_training parameters for clean model training for consistency ---
    train_config = config['victim_training']

    # Load data
    train_loader = get_dataloader(train_config['batch_size'], True, config['dataset']['path'], config['num_workers'])
    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # Initialize a clean model
    model = ResNet18(num_classes=config['dataset']['num_classes']).to(device)

    # Setup optimizer and loss
    optimizer = optim.SGD(model.parameters(), lr=train_config['lr'], momentum=train_config['momentum'],
                          weight_decay=train_config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['epochs'])
    criterion = nn.CrossEntropyLoss()

    print("--- Starting Clean Surrogate Model Training ---")
    best_acc = 0.0

    for epoch in range(train_config['epochs']):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_config['epochs']}")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{running_loss / (progress_bar.n + 1):.4f}'})

        scheduler.step()

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = 100 * correct / total
        print(f"\nEpoch {epoch + 1} Test Accuracy: {acc:.2f}%")

        # Save the best model
        if acc > best_acc:
            best_acc = acc
            save_dir = './pretrained'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'resnet18_cifar10.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path} with accuracy {best_acc:.2f}%")

    print(f"--- Finished Clean Surrogate Model Training ---")
    print(f"Final best model is saved at './pretrained/resnet18_cifar10.pth' with accuracy {best_acc:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file')
    args = parser.parse_args()
    main(args.config)
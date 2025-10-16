#
# 请用以下全部代码覆盖您的 train_clean_model.py 文件
#
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

# --- 环境设置 ---
project_dir = os.path.dirname(os.path.abspath(__file__))
torch_hub_dir = os.path.join(project_dir, 'pretrained', 'torch_hub')
os.makedirs(torch_hub_dir, exist_ok=True)
torch.hub.set_dir(torch_hub_dir)
# --- 环境设置结束 ---

from core.models.resnet import ResNet18
from data.cifar10 import get_dataloader


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 设备设置 ---
    if torch.cuda.is_available() and config.get('device_ids'):
        device_ids = config['device_ids']
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
        if len(device_ids) > 0:
            device = torch.device("cuda:0")
            print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            device_ids = []
            device = torch.device("cpu")
            print("No GPUs specified, using CPU.")
    else:
        device_ids = []
        device = torch.device("cpu")
        print("CUDA not available or no GPUs specified, using CPU.")
    # --- 设备设置结束 ---

    torch.manual_seed(config['seed'])

    train_config = config['victim_training']

    train_loader = get_dataloader(train_config['batch_size'], True, config['dataset']['path'], config['num_workers'])
    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # --- 模型初始化 ---
    model = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model)

    params_to_optimize = model.module.parameters() if isinstance(model, nn.DataParallel) else model.parameters()
    optimizer = optim.SGD(params_to_optimize, lr=train_config['lr'], momentum=train_config['momentum'],
                          weight_decay=train_config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['epochs'])
    criterion = nn.CrossEntropyLoss()

    print("--- Starting Clean Surrogate Model Training ---")
    best_acc = 0.0

    for epoch in range(train_config['epochs']):
        model_to_train = model.module if isinstance(model, nn.DataParallel) else model
        model_to_train.train()

        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_config['epochs']}")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)

            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{running_loss / (progress_bar.n + 1):.4f}'})

        scheduler.step()

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

        if acc > best_acc:
            best_acc = acc
            save_dir = './pretrained'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'resnet18_cifar10.pth')

            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), save_path)

            print(f"Best model saved to {save_path} with accuracy {best_acc:.2f}%")

    print(f"--- Finished Clean Surrogate Model Training ---")
    print(f"Final best model is saved at './pretrained/resnet18_cifar10.pth' with accuracy {best_acc:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file')
    args = parser.parse_args()
    main(args.config)
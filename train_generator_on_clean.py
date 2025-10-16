import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse

from core.models.resnet import ResNet18
from core.models.generator import TriggerNet
from data.cifar10 import get_dataloader
from core.attack import BackdoorAttack


# --- 新增辅助函数 ---
def get_target_features(classifier, data_loader, target_class, device):
    """
    预计算目标类别的平均特征向量。
    """
    classifier.eval()
    target_features_list = []
    print(f"Pre-calculating target features for class {target_class}...")
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            target_mask = (y == target_class)
            if target_mask.any():
                x_target = x[target_mask]
                if isinstance(classifier, nn.DataParallel):
                    features = classifier.module.get_features(x_target)
                else:
                    features = classifier.get_features(x_target)
                target_features_list.append(features.cpu())

    if not target_features_list:
        raise ValueError(f"No samples found for target class {target_class} in the provided data.")

    all_target_features = torch.cat(target_features_list, dim=0)
    mean_target_features = all_target_features.mean(dim=0, keepdim=True).to(device)
    print("Target features pre-calculated successfully.")
    return mean_target_features


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['seed'])

    gen_config = config['generator_training']

    train_loader = get_dataloader(gen_config['batch_size'], True, config['dataset']['path'], config['num_workers'])

    classifier = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    classifier.load_state_dict(torch.load(config['model']['clean_model_path'], map_location=device))
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    print(f"Loaded and froze clean classifier from {config['model']['clean_model_path']}")

    # --- 核心修改: 预计算目标特征 ---
    target_features = get_target_features(classifier, train_loader, gen_config['target_class'], device)

    generator = TriggerNet().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=gen_config['optimizer_G']['lr'])
    attack_helper = BackdoorAttack(generator, classifier, gen_config, device)

    print("\n--- Stage 1: Training Generator with Feature Matching ---")

    for epoch in range(gen_config['epochs']):
        generator.train()
        progress_bar = tqdm(train_loader, desc=f"Generator Training Epoch {epoch + 1}/{gen_config['epochs']}")
        total_loss_G = 0.0

        for x_clean, y_true in progress_bar:
            x_clean, y_true = x_clean.to(device), y_true.to(device)
            non_target_mask = (y_true != gen_config['target_class'])
            if not non_target_mask.any(): continue

            x_for_G = x_clean[non_target_mask]

            x_poisoned_for_G, delta_patch = attack_helper.generate_poisoned_sample(x_for_G)

            # --- 核心修改: 传入目标特征 ---
            loss_G = attack_helper.calculate_generator_loss(x_for_G, x_poisoned_for_G, delta_patch, target_features)

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            total_loss_G += loss_G.item()
            progress_bar.set_postfix({'Loss_G': f'{loss_G.item():.4f}'})

        print(f"Epoch {epoch + 1} average Loss_G: {total_loss_G / len(train_loader):.4f}")

    save_dir = gen_config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(save_dir, gen_config['generator_save_name']))
    print(f"\nFinished Stage 1. Best generator saved to {os.path.join(save_dir, gen_config['generator_save_name'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Train trigger generator on a fixed clean model.")
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)
# =================================================================================================
# train_advanced.py (Import修复 & 最终版)
# =================================================================================================
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml
import argparse
import numpy as np

# [修复] 只从 data.cifar10 导入我们需要的 get_dataloader 函数
from data.cifar10 import get_dataloader
from core.models.resnet import ResNet18
from core.models.generator import MultiScaleAttentionGenerator
from core.attack import AdversarialColearningAttack


def train(cfg):
    # --- 0. Setup GPU Environment from Config ---
    gpu_ids = str(cfg['train']['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_list = [int(i) for i in gpu_ids.split(',')]
    num_gpus = len(gpu_list)
    print(f"Using device: {device}, GPU IDs from config: '{gpu_ids}', Number of GPUs: {num_gpus}")

    # Create directories
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['model_dir'], exist_ok=True)

    # --- 1. Load Datasets and Prepare Data Samplers ---
    # 注意：我们直接使用 get_dataloader 函数，不需要自己处理 Dataset 对象
    target_class_idx = cfg['attack']['target_class_idx']

    # 为了能够按类别采样，我们仍然需要先拿到dataset对象
    train_dataset = get_dataloader(cfg['dataset']['batch_size'], train=True, path=cfg['dataset']['root_dir']).dataset

    # Adjust batch size for multi-GPU training
    batch_size = cfg['dataset']['batch_size'] * num_gpus
    print(f"Effective batch size: {batch_size}")

    # Create indices for target and non-target classes
    target_indices = [i for i, label in enumerate(train_dataset.targets) if label == target_class_idx]
    nontarget_indices = [i for i, label in enumerate(train_dataset.targets) if label != target_class_idx]

    # Create separate DataLoaders for each track's needs
    victim_poison_loader = DataLoader(Subset(train_dataset, target_indices),
                                      batch_size=int(batch_size * cfg['attack']['poisoning_ratio']), shuffle=True,
                                      drop_last=True, num_workers=cfg['dataset']['num_workers'])
    victim_clean_loader = DataLoader(Subset(train_dataset, nontarget_indices),
                                     batch_size=int(batch_size * (1 - cfg['attack']['poisoning_ratio'])), shuffle=True,
                                     drop_last=True, num_workers=cfg['dataset']['num_workers'])
    generator_source_loader = DataLoader(Subset(train_dataset, nontarget_indices), batch_size=batch_size, shuffle=True,
                                         drop_last=True, num_workers=cfg['dataset']['num_workers'])

    # --- 2. Build All Three Models ---
    victim_model = ResNet18(num_classes=cfg['dataset']['num_classes']).to(device)
    generator = MultiScaleAttentionGenerator(input_channels=3).to(device)
    teacher_model = ResNet18(num_classes=cfg['dataset']['num_classes']).to(device)

    print(f"Loading pre-trained teacher model from: {cfg['teacher_model_path']}")
    teacher_model.load_state_dict(torch.load(cfg['teacher_model_path'], map_location=device))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # --- Wrap models for multi-GPU training if specified ---
    if num_gpus > 1:
        print("Using DataParallel for multi-GPU training.")
        victim_model = nn.DataParallel(victim_model, device_ids=gpu_list)
        generator = nn.DataParallel(generator, device_ids=gpu_list)
        teacher_model = nn.DataParallel(teacher_model, device_ids=gpu_list)

    # --- 3. Setup Optimizers and Loss ---
    optimizer_V = torch.optim.Adam(victim_model.parameters(), lr=cfg['optimizer']['lr_F'],
                                   betas=(cfg['optimizer']['beta1'], cfg['optimizer']['beta2']))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg['optimizer']['lr_G'],
                                   betas=(cfg['optimizer']['beta1'], cfg['optimizer']['beta2']))
    criterion_ce = nn.CrossEntropyLoss()

    # --- 4. Create Attack Helper ---
    generator_module = generator.module if isinstance(generator, nn.DataParallel) else generator
    attack_helper = AdversarialColearningAttack(cfg, generator_module, criterion_ce, device)

    # --- 5. The Training Loop ---
    max_epochs = cfg['train']['max_epochs']
    print("Starting Adversarially-Guided Backdoor Co-learning...")

    for epoch in range(max_epochs):
        victim_poison_iter = iter(victim_poison_loader)
        victim_clean_iter = iter(victim_clean_loader)
        generator_iter = iter(generator_source_loader)

        num_batches = min(len(victim_poison_loader), len(victim_clean_loader), len(generator_source_loader))
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{max_epochs}")

        for i in progress_bar:
            # --- Victim Learning Track ---
            victim_model.train()
            generator.eval()

            x_target_source, y_target_source = next(victim_poison_iter)
            x_clean, y_clean = next(victim_clean_iter)

            x_target_source, y_target_source = x_target_source.to(device), y_target_source.to(device)
            x_clean, y_clean = x_clean.to(device), y_clean.to(device)

            x_poisoned_target = attack_helper.generate_poisoned_sample(x_target_source, is_train=False)

            x_combined = torch.cat([x_clean, x_poisoned_target.detach()], dim=0)
            y_combined = cat([y_clean, y_target_source], dim=0)

            outputs = victim_model(x_combined)
            loss_victim = criterion_ce(outputs, y_combined)

            optimizer_V.zero_grad()
            loss_victim.backward()
            optimizer_V.step()

            # --- Generator Forging Track ---
            victim_model.eval()
            generator.train()

            x_nontarget_source, _ = next(generator_iter)
            x_nontarget_source = x_nontarget_source.to(device)

            x_poisoned_nontarget = attack_helper.generate_poisoned_sample(x_nontarget_source, is_train=True)

            loss_generator = attack_helper.calculate_generator_loss(
                x_nontarget_source, x_poisoned_nontarget, victim_model, teacher_model
            )

            optimizer_G.zero_grad()
            loss_generator.backward()
            optimizer_G.step()

            progress_bar.set_postfix(
                Loss_Victim=f"{loss_victim.item():.4f}",
                Loss_Generator=f"{loss_generator.item():.4f}"
            )

        # --- Save model checkpoints correctly ---
        victim_state_dict = victim_model.module.state_dict() if isinstance(victim_model,
                                                                           nn.DataParallel) else victim_model.state_dict()
        generator_state_dict = generator.module.state_dict() if isinstance(generator,
                                                                           nn.DataParallel) else generator.state_dict()

        torch.save(victim_state_dict, os.path.join(cfg['model_dir'], f'victim_model_epoch_{epoch + 1}.pth'))
        torch.save(generator_state_dict, os.path.join(cfg['model_dir'], f'generator_epoch_{epoch + 1}.pth'))
        print(f"Epoch {epoch + 1} finished. Models saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adversarially-Guided Backdoor Co-learning")
    parser.add_argument('--config', type=str, default='configs/cifar10_advanced_joint.yml',
                        help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    train(cfg)
# =================================================================================================
# evaluate_victim.py (Import修复 & 最终版)
# =================================================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import numpy as np

# [修复] 从 data.cifar10 导入正确的 get_dataloader 函数
from data.cifar10 import get_dataloader
from core.models.resnet import ResNet18
from core.models.generator import MultiScaleAttentionGenerator
from core.attack import AdversarialColearningAttack


def evaluate(cfg):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CIFAR-10 test dataset using the correct function
    test_loader = get_dataloader(
        batch_size=cfg['dataset']['batch_size'],
        train=False,
        path=cfg['dataset']['root_dir'],
        num_workers=cfg['dataset']['num_workers']
    )

    target_class_idx = cfg['attack']['target_class_idx']

    # --- 1. Load Models ---
    victim_model = ResNet18(num_classes=cfg['dataset']['num_classes']).to(device)
    victim_model_path = cfg['eval']['victim_model_path']
    print(f"Loading victim model from: {victim_model_path}")
    victim_model.load_state_dict(torch.load(victim_model_path, map_location=device))
    victim_model.eval()

    generator = MultiScaleAttentionGenerator(input_channels=3).to(device)
    generator_path = cfg['eval']['generator_path']
    print(f"Loading generator from: {generator_path}")
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    # --- 2. Setup Attack Helper ---
    criterion_ce = nn.CrossEntropyLoss()
    attack_helper = AdversarialColearningAttack(cfg, generator, criterion_ce, device)

    # --- 3. Evaluation ---
    total_correct_clean = 0
    total_correct_poisoned = 0
    total_poisoned_samples = 0
    total_samples = 0

    progress_bar = tqdm(test_loader, desc="Evaluating")
    for x_batch, y_batch in progress_bar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # a) Evaluate Benign Accuracy (BA)
        with torch.no_grad():
            outputs_clean = victim_model(x_batch)
            _, predicted_clean = torch.max(outputs_clean.data, 1)
            total_correct_clean += (predicted_clean == y_batch).sum().item()
        total_samples += y_batch.size(0)

        # b) Evaluate Attack Success Rate (ASR)
        nontarget_mask = (y_batch != target_class_idx)
        if nontarget_mask.sum() > 0:
            x_source = x_batch[nontarget_mask]
            x_poisoned = attack_helper.generate_poisoned_sample(x_source, is_train=False)
            with torch.no_grad():
                outputs_poisoned = victim_model(x_poisoned)
                _, predicted_poisoned = torch.max(outputs_poisoned.data, 1)
                total_correct_poisoned += (predicted_poisoned == target_class_idx).sum().item()
            total_poisoned_samples += x_source.size(0)

    benign_accuracy = 100 * total_correct_clean / total_samples
    attack_success_rate = 100 * total_correct_poisoned / total_poisoned_samples

    print(f"Evaluation Results:")
    print(f"  Benign Accuracy (BA): {benign_accuracy:.2f}%")
    print(f"  Attack Success Rate (ASR): {attack_success_rate:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Backdoored Victim Model")
    parser.add_argument('--config', type=str, default='configs/cifar10_advanced_joint.yml',
                        help='Path to the config file')
    parser.add_argument('--victim_model_path', type=str, required=True,
                        help='Path to the trained victim model checkpoint')
    parser.add_argument('--generator_path', type=str, required=True, help='Path to the trained generator checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['eval'] = {
        'victim_model_path': args.victim_model_path,
        'generator_path': args.generator_path
    }

    evaluate(cfg)
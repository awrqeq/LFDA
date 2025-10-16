import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import lpips

# --- 环境设置 ---
project_dir = os.path.dirname(os.path.abspath(__file__))
torch_hub_dir = os.path.join(project_dir, 'pretrained', 'torch_hub')
os.makedirs(torch_hub_dir, exist_ok=True)
torch.hub.set_dir(torch_hub_dir)
# --- 环境设置结束 ---

from core.models.resnet import ResNet18
from core.models.generator import MultiScaleAttentionGenerator
from data.cifar10 import get_dataloader
from core.attack import BackdoorAttack


# --- 全新的、可复用的评估函数 ---
def evaluate_advanced(classifier, generator, attack_helper, test_loader, device):
    classifier.eval()
    generator.eval()
    total_samples, clean_correct, poisoned_correct, poisoned_total = 0, 0, 0, 0

    with torch.no_grad():
        for x_clean, y_true in tqdm(test_loader, desc="Evaluating"):
            x_clean, y_true = x_clean.to(device), y_true.to(device)

            # 评估BA
            outputs_clean = classifier(x_clean)
            _, predicted_clean = torch.max(outputs_clean, 1)
            total_samples += y_true.size(0)
            clean_correct += (predicted_clean == y_true).sum().item()

            # 评估ASR
            non_target_mask = (y_true != attack_helper.target_class)
            if non_target_mask.any():
                x_to_poison = x_clean[non_target_mask]
                # 使用 is_train=False 来激活“强触发”
                x_poisoned = attack_helper.generate_poisoned_sample(x_to_poison, is_train=False)

                outputs_poisoned = classifier(x_poisoned)
                _, predicted_poisoned = torch.max(outputs_poisoned, 1)

                poisoned_total += x_to_poison.size(0)
                poisoned_correct += (predicted_poisoned == attack_helper.target_class).sum().item()

    ba = 100 * clean_correct / total_samples
    asr = 100 * poisoned_correct / poisoned_total if poisoned_total > 0 else 0
    return ba, asr


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['seed'])
    train_config = config['joint_training']

    train_loader = get_dataloader(train_config['batch_size'], True, config['dataset']['path'], config['num_workers'])
    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    classifier = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    generator = MultiScaleAttentionGenerator().to(device)

    optimizer_F = optim.SGD(classifier.parameters(), lr=train_config['optimizer_F']['lr'],
                            momentum=train_config['optimizer_F']['momentum'],
                            weight_decay=train_config['optimizer_F']['weight_decay'])
    optimizer_G = optim.Adam(generator.parameters(), lr=train_config['optimizer_G']['lr'])
    scheduler_F = optim.lr_scheduler.CosineAnnealingLR(optimizer_F, T_max=train_config['epochs'])

    criterion_ce = nn.CrossEntropyLoss()
    attack_helper = BackdoorAttack(generator, classifier, config, device)

    print("\n--- Starting Advanced Joint End-to-End Training ---")

    for epoch in range(train_config['epochs']):
        classifier.train()
        generator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_config['epochs']}")

        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            split_idx = int(train_config['poison_ratio_in_batch'] * x_batch.size(0))
            x_clean, y_clean = x_batch[split_idx:], y_batch[split_idx:]
            x_source, _ = x_batch[:split_idx], y_batch[:split_idx]

            # --- 1. 良性路径 ---
            optimizer_F.zero_grad()
            if x_clean.size(0) > 0:
                outputs_clean = classifier(x_clean)
                loss_benign = criterion_ce(outputs_clean, y_clean)
                loss_benign.backward()
                optimizer_F.step()
            else:
                loss_benign = torch.tensor(0.0)

            # --- 2. 后门路径 ---
            if x_source.size(0) > 0:
                # 使用 is_train=True 来激活“弱触发”和“动态强度”
                x_poisoned = attack_helper.generate_poisoned_sample(x_source, is_train=True)
                loss_backdoor = attack_helper.calculate_joint_backdoor_loss(x_source, x_poisoned)

                optimizer_F.zero_grad()
                optimizer_G.zero_grad()
                loss_backdoor.backward()
                optimizer_F.step()
                optimizer_G.step()
            else:
                loss_backdoor = torch.tensor(0.0)

            progress_bar.set_postfix(
                {'L_Benign': f'{loss_benign.item():.4f}', 'L_Backdoor': f'{loss_backdoor.item():.4f}'})

        scheduler_F.step()

        if (epoch + 1) % train_config['eval_every_epochs'] == 0 or (epoch + 1) == train_config['epochs']:
            ba, asr = evaluate_advanced(classifier, generator, attack_helper, test_loader, device)
            print(f"\nEpoch {epoch + 1}: BA = {ba:.2f}%, ASR = {asr:.2f}%")

    save_dir = train_config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    torch.save(classifier.state_dict(), os.path.join(save_dir, train_config['classifier_save_name']))
    torch.save(generator.state_dict(), os.path.join(save_dir, train_config['generator_save_name']))
    print(f"\nFinal models saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Joint Training for Backdoor Attacks.")
    parser.add_argument('--config', type=str, default='configs/cifar10_advanced_joint.yml',
                        help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)
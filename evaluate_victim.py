#
# 请用以下全部代码覆盖您的 evaluate_victim.py 文件
#
import yaml
import torch
import torch.nn as nn
import os
from tqdm import tqdm

# --- 环境设置：确保 torch.hub 下载到项目内部 ---
project_dir = os.path.dirname(os.path.abspath(__file__))
torch_hub_dir = os.path.join(project_dir, 'pretrained', 'torch_hub')
os.makedirs(torch_hub_dir, exist_ok=True)
torch.hub.set_dir(torch_hub_dir)
# --- 环境设置结束 ---

from core.models.resnet import ResNet18
from core.models.generator import TriggerNet
from data.cifar10 import get_dataloader
from core.utils import freq_utils


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 设备设置，支持多GPU ---
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

    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # --- 模型加载 ---
    victim_model = ResNet18(num_classes=config['dataset']['num_classes'])
    victim_model.load_state_dict(torch.load(config['evaluation']['victim_model_path'], map_location='cpu'))
    victim_model = victim_model.to(device)

    generator = TriggerNet()
    generator.load_state_dict(torch.load(config['evaluation']['generator_path'], map_location='cpu'))
    generator = generator.to(device)

    if len(device_ids) > 1:
        victim_model = nn.DataParallel(victim_model)
        generator = nn.DataParallel(generator)

    victim_model.eval()
    generator.eval()

    print("--- Starting Final Evaluation ---")

    # --- 获取NFF参数 ---
    trigger_net_config = config['generator_training']['trigger_net']
    injection_strength = config['generator_training']['injection_strength']
    target_class = config['victim_training']['target_class']

    total_correct_clean, total_correct_poisoned, total_poisoned_samples, total_samples = 0, 0, 0, 0

    with torch.no_grad():
        print("Evaluating Benign Accuracy (BA)...")
        for x, y in tqdm(test_loader, desc="BA"):
            x, y = x.to(device), y.to(device)
            outputs = victim_model(x)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += y.size(0)
            total_correct_clean += (predicted == y).sum().item()
        ba = 100 * total_correct_clean / total_samples if total_samples > 0 else 0

        print("Evaluating Attack Success Rate (ASR)...")
        for x, y in tqdm(test_loader, desc="ASR"):
            x, y = x.to(device), y.to(device)
            non_target_mask = (y != target_class)
            if not non_target_mask.any():
                continue

            x_to_poison = x[non_target_mask]

            # --- 使用NFF逻辑生成毒化样本 ---
            x_freq = freq_utils.to_freq(x_to_poison)
            freq_patch = freq_utils.extract_freq_patch_and_reshape(x_freq, **trigger_net_config)
            delta_patch = generator(freq_patch)
            poisoned_freq = freq_utils.reshape_and_insert_freq_patch(x_freq, delta_patch, strength=injection_strength,
                                                                     **trigger_net_config)
            x_p = freq_utils.to_spatial(poisoned_freq)
            x_p = torch.clamp(x_p, 0, 1)

            outputs = victim_model(x_p)
            _, predicted = torch.max(outputs.data, 1)

            total_poisoned_samples += x_to_poison.size(0)
            total_correct_poisoned += (predicted == target_class).sum().item()

    asr = 100 * total_correct_poisoned / total_poisoned_samples if total_poisoned_samples > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Benign Accuracy (BA): {ba:.2f}%")
    print(f"Attack Success Rate (ASR): {asr:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file')
    args = parser.parse_args()
    main(args.config)
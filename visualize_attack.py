#
# 请用以下全部代码覆盖您的 visualize_attack.py 文件
#
import yaml
import torch
import torch.nn as nn
import os

# --- 环境设置 ---
project_dir = os.path.dirname(os.path.abspath(__file__))
torch_hub_dir = os.path.join(project_dir, 'pretrained', 'torch_hub')
os.makedirs(torch_hub_dir, exist_ok=True)
torch.hub.set_dir(torch_hub_dir)
# --- 环境设置结束 ---

from data.cifar10 import get_dataloader
from core.models.generator import TriggerNet
from core.utils.image_utils import save_image_grid
from core.utils import freq_utils


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

    test_loader = get_dataloader(batch_size=8, train=False, path=config['dataset']['path'])

    # --- 模型加载 ---
    generator = TriggerNet()
    generator.load_state_dict(torch.load(config['evaluation']['generator_path'], map_location='cpu'))
    generator = generator.to(device)

    if len(device_ids) > 1:
        generator = nn.DataParallel(generator)

    generator.eval()

    x, y = next(iter(test_loader))
    x = x.to(device)

    # --- 获取NFF参数 ---
    trigger_net_config = config['generator_training']['trigger_net']
    injection_strength = config['generator_training']['injection_strength']

    # --- 使用NFF逻辑生成毒化样本 ---
    with torch.no_grad():
        x_freq = freq_utils.to_freq(x)
        freq_patch = freq_utils.extract_freq_patch_and_reshape(x_freq, **trigger_net_config)
        delta_patch = generator(freq_patch)
        poisoned_freq = freq_utils.reshape_and_insert_freq_patch(x_freq, delta_patch, strength=injection_strength,
                                                                 **trigger_net_config)
        x_p = freq_utils.to_spatial(poisoned_freq)
        x_p = torch.clamp(x_p, 0, 1)

    residual = torch.abs(x - x_p)
    residual_amplified = torch.clamp(residual * 20, 0, 1)  # Amplify more for better visibility

    images_to_save, labels = [], []
    for i in range(x.size(0)):
        images_to_save.extend([x[i], x_p[i], residual_amplified[i]])
        labels.extend([f'Original #{i}', f'Poisoned #{i}', f'Residual x20 #{i}'])

    save_dir = './outputs'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'nff_attack_visualization.png')

    save_image_grid(images_to_save, labels, save_path, nrow=3)
    print(f"NFF Visualization saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file')
    args = parser.parse_args()
    main(args.config)
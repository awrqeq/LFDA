# =================================================================================================
# visualize_attack.py (Import修复 & 最终版)
# =================================================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import matplotlib.pyplot as plt

# [修复] 从 data.cifar10 导入正确的 get_dataloader 函数
from data.cifar10 import get_dataloader
from core.models.generator import MultiScaleAttentionGenerator
from core.attack import AdversarialColearningAttack
from core.utils import image_utils


def visualize(cfg):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CIFAR-10 test dataset using the correct function
    test_loader = get_dataloader(
        batch_size=cfg['visualize']['num_images'],
        train=False,
        path=cfg['dataset']['root_dir'],
        num_workers=cfg['dataset']['num_workers']
    )

    # Load the generator
    generator = MultiScaleAttentionGenerator(input_channels=3).to(device)
    generator_path = cfg['visualize']['generator_path']
    print(f"Loading generator from: {generator_path}")
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    # Setup Attack Helper
    criterion_ce = nn.CrossEntropyLoss()
    attack_helper = AdversarialColearningAttack(cfg, generator, criterion_ce, device)

    # Get a batch of images for visualization
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch.to(device)

    # Generate poisoned samples
    x_poisoned = attack_helper.generate_poisoned_sample(x_batch, is_train=False)

    # Calculate the perturbation (trigger)
    perturbation = x_poisoned - x_batch

    # Convert tensors to displayable format
    x_clean_vis = image_utils.unnormalize_batch(x_batch.cpu())
    x_poisoned_vis = image_utils.unnormalize_batch(x_poisoned.cpu())
    perturbation_vis = image_utils.normalize_batch(perturbation.cpu())

    # Create visualization
    num_images = cfg['visualize']['num_images']
    fig, axs = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
    fig.suptitle("Attack Visualization", fontsize=16)

    for i in range(num_images):
        axs[0, i].imshow(x_clean_vis[i].permute(1, 2, 0))
        axs[0, i].set_title(f"Clean #{i + 1}")
        axs[0, i].axis('off')

        axs[1, i].imshow(perturbation_vis[i].permute(1, 2, 0))
        axs[1, i].set_title("Perturbation")
        axs[1, i].axis('off')

        axs[2, i].imshow(x_poisoned_vis[i].permute(1, 2, 0))
        axs[2, i].set_title("Poisoned")
        axs[2, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    save_path = cfg['visualize']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Backdoor Attack")
    parser.add_argument('--config', type=str, default='configs/cifar10_advanced_joint.yml',
                        help='Path to the config file')
    parser.add_argument('--generator_path', type=str, required=True, help='Path to the trained generator checkpoint')
    parser.add_argument('--num_images', type=int, default=8, help='Number of images to visualize')
    parser.add_argument('--save_path', type=str, default='visualizations/attack_visualization.png',
                        help='Path to save the output image')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['visualize'] = {
        'generator_path': args.generator_path,
        'num_images': args.num_images,
        'save_path': args.save_path
    }

    visualize(cfg)
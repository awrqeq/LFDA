import yaml
import torch
import os
from data.cifar10 import get_dataloader
from core.models.generator import UNet
from core.utils.image_utils import save_image_grid


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])

    test_loader = get_dataloader(batch_size=8, train=False, path=config['dataset']['path'])  # Get a small batch

    generator = UNet().to(device)
    generator.load_state_dict(torch.load(config['evaluation']['generator_path'], map_location=device))
    generator.eval()

    # Get a batch of images
    x, y = next(iter(test_loader))
    x = x.to(device)

    # Generate poisoned versions
    with torch.no_grad():
        delta_phi = generator(x)
        x_freq = torch.fft.fft2(x, dim=(-2, -1))
        amp, phase = x_freq.abs(), x_freq.angle()
        poisoned_phase = phase + delta_phi
        poisoned_freq = torch.polar(amp, poisoned_phase)
        x_p = torch.fft.ifft2(poisoned_freq, dim=(-2, -1)).real
        x_p = torch.clamp(x_p, 0, 1)

    # Calculate residual
    residual = torch.abs(x - x_p)
    residual_amplified = torch.clamp(residual * 10, 0, 1)  # Amplify for visibility

    # Prepare for saving
    images_to_save = []
    labels = []
    for i in range(x.size(0)):
        images_to_save.extend([x[i], x_p[i], residual_amplified[i]])
        labels.extend([f'Original #{i}', f'Poisoned #{i}', f'Residual x10 #{i}'])

    save_dir = './outputs'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'attack_visualization.png')

    save_image_grid(images_to_save, labels, save_path, nrow=3)
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file')
    args = parser.parse_args()
    main(args.config)

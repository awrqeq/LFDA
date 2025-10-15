import yaml
import torch
import os
from tqdm import tqdm

from core.models.resnet import ResNet18
from core.models.generator import UNet
from data.cifar10 import get_dataloader


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])

    test_loader = get_dataloader(config['evaluation']['batch_size'], False, config['dataset']['path'],
                                 config['num_workers'])

    # Load models
    victim_model = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    victim_model.load_state_dict(torch.load(config['evaluation']['victim_model_path'], map_location=device))

    generator = UNet().to(device)
    generator.load_state_dict(torch.load(config['evaluation']['generator_path'], map_location=device))

    victim_model.eval()
    generator.eval()

    print("--- Starting Final Evaluation ---")

    total_correct_clean = 0
    total_correct_poisoned = 0
    total_poisoned_samples = 0
    total_samples = 0

    with torch.no_grad():
        # Evaluate Benign Accuracy (BA)
        print("Evaluating Benign Accuracy (BA)...")
        for x, y in tqdm(test_loader, desc="BA"):
            x, y = x.to(device), y.to(device)
            outputs = victim_model(x)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += y.size(0)
            total_correct_clean += (predicted == y).sum().item()

        ba = 100 * total_correct_clean / total_samples

        # Evaluate Attack Success Rate (ASR)
        print("Evaluating Attack Success Rate (ASR)...")
        for x, y in tqdm(test_loader, desc="ASR"):
            x, y = x.to(device), y.to(device)
            non_target_mask = (y != config['victim_training']['target_class'])
            if not non_target_mask.any():
                continue

            x_to_poison = x[non_target_mask]

            # Generate poisoned images
            delta_phi = generator(x_to_poison)
            x_freq = torch.fft.fft2(x_to_poison, dim=(-2, -1))
            amp, phase = x_freq.abs(), x_freq.angle()
            poisoned_phase = phase + delta_phi
            poisoned_freq = torch.polar(amp, poisoned_phase)
            x_p = torch.fft.ifft2(poisoned_freq, dim=(-2, -1)).real
            x_p = torch.clamp(x_p, 0, 1)

            outputs = victim_model(x_p)
            _, predicted = torch.max(outputs.data, 1)

            total_poisoned_samples += x_to_poison.size(0)
            total_correct_poisoned += (predicted == config['victim_training']['target_class']).sum().item()

    asr = 100 * total_correct_poisoned / total_poisoned_samples

    print("\n--- Evaluation Results ---")
    print(f"Benign Accuracy (BA): {ba:.2f}%")
    print(f"Attack Success Rate (ASR): {asr:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar10_resnet18.yml', help='Path to the config file')
    args = parser.parse_args()
    main(args.config)

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import argparse

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


def evaluate_advanced(classifier, generator, attack_helper, test_loader, device):
    """
    评估后门模型的性能（高级版本）。

    Args:
        classifier (nn.Module): 待评估的分类器模型。
        generator (nn.Module): 触发器生成器。
        attack_helper (BackdoorAttack): 攻击助手实例，用于调用其方法和参数。
        test_loader (DataLoader): 测试数据集的加载器。
        device (torch.device): 运行设备 (CPU or GPU)。

    Returns:
        tuple[float, float]: (良性准确率 BA, 攻击成功率 ASR)，均为百分比。
    """
    classifier.eval()
    generator.eval()

    total_samples = 0
    clean_correct = 0
    poisoned_correct_as_target = 0
    poisoned_total = 0

    progress_bar = tqdm(test_loader, desc="Evaluating")

    with torch.no_grad():
        for x_clean, y_true in progress_bar:
            x_clean, y_true = x_clean.to(device), y_true.to(device)

            # --- 评估良性准确率 (BA) ---
            outputs_clean = classifier(x_clean)
            _, predicted_clean = torch.max(outputs_clean, 1)
            total_samples += y_true.size(0)
            clean_correct += (predicted_clean == y_true).sum().item()

            # --- 评估攻击成功率 (ASR) ---
            # 只对非目标类别的样本进行攻击测试
            non_target_mask = (y_true != attack_helper.target_class)
            if non_target_mask.any():
                x_to_poison = x_clean[non_target_mask]

                # 使用 is_train=False 来激活在评估时使用的“强触发”
                x_poisoned = attack_helper.generate_poisoned_sample(x_to_poison, is_train=False)

                outputs_poisoned = classifier(x_poisoned)
                _, predicted_poisoned = torch.max(outputs_poisoned, 1)

                poisoned_total += x_to_poison.size(0)
                poisoned_correct_as_target += (predicted_poisoned == attack_helper.target_class).sum().item()

            ba_running = 100 * clean_correct / total_samples if total_samples > 0 else 0
            asr_running = 100 * poisoned_correct_as_target / poisoned_total if poisoned_total > 0 else 0
            progress_bar.set_postfix({
                'BA': f'{ba_running:.2f}%',
                'ASR': f'{asr_running:.2f}%'
            })

    ba = 100 * clean_correct / total_samples
    asr = 100 * poisoned_correct_as_target / poisoned_total if poisoned_total > 0 else 0

    return ba, asr


def main(config_path):
    """
    主函数，用于独立运行评估脚本。
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 设备设置 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['seed'])
    eval_config = config['evaluation']

    # --- 数据加载器 ---
    test_loader = get_dataloader(eval_config['batch_size'], False, config['dataset']['path'], config['num_workers'])

    # --- 模型加载 ---
    # 分类器
    victim_classifier = ResNet18(num_classes=config['dataset']['num_classes']).to(device)
    victim_classifier.load_state_dict(torch.load(eval_config['victim_model_path'], map_location=device))
    print(f"Loaded victim model from: {eval_config['victim_model_path']}")

    # 生成器
    generator = MultiScaleAttentionGenerator().to(device)
    generator.load_state_dict(torch.load(eval_config['generator_path'], map_location=device))
    print(f"Loaded generator from: {eval_config['generator_path']}")

    # 创建一个临时的 attack_helper 实例以传递给评估函数
    attack_helper = BackdoorAttack(generator, victim_classifier, config, device)

    # --- 执行评估 ---
    ba, asr = evaluate_advanced(victim_classifier, generator, attack_helper, test_loader, device)

    print("\n--- Final Evaluation Results ---")
    print(f"Benign Accuracy (BA): {ba:.2f}%")
    print(f"Attack Success Rate (ASR): {asr:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a backdoored victim model.")
    # 让配置文件路径更灵活
    parser.add_argument('--config', type=str, default='configs/cifar10_advanced_joint.yml',
                        help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)
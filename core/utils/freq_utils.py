import torch
from pytorch_wavelets import DWTForward, DWTInverse
from torch_dct import dct_2d, idct_2d


# --- 原有的 FFT 函数保持不变 ---
def to_freq(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(x, norm="ortho")


def to_spatial(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(x, norm="ortho").real


def extract_freq_patch_and_reshape(x_freq, patch_size, start_row, start_col):
    patch_freq = x_freq[:, :, start_row: start_row + patch_size, start_col: start_col + patch_size]
    return patch_freq.reshape(patch_freq.size(0), -1)


def reshape_and_insert_freq_patch(x_freq, delta_patch_reshaped, strength, patch_size, start_row, start_col):
    delta_patch = delta_patch_reshaped.reshape(
        delta_patch_reshaped.size(0), 3, patch_size, patch_size
    )
    x_freq[:, :, start_row: start_row + patch_size, start_col: start_col + patch_size] += (
            strength * delta_patch
    )
    return x_freq


# --- 全新的、修正后的 DWT/IDWT 函数 ---

def get_dwt_idwt(device):
    """获取DWT和IDWT变换器"""
    # J=1 代表进行一级小波分解
    dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)
    idwt = DWTInverse(wave='haar', mode='zero').to(device)
    return dwt, idwt


def to_dwt(x: torch.Tensor, dwt: DWTForward) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    执行DWT变换，并正确解包。
    返回:
        ll (torch.Tensor): 低频子带。
        high_freqs (list[torch.Tensor]): 一个包含 [lh, hl, hh] 三个独立张量的列表。
    """
    ll, yh = dwt(x)
    # yh 是一个列表，对于 J=1，它只包含一个张量 yh[0]
    # yh[0] 的形状是 (N, C, 3, H, W)，其中第3个维度堆叠了 lh, hl, hh
    # 我们需要将它们解包成三个独立的张量
    lh = yh[0][:, :, 0, :, :]
    hl = yh[0][:, :, 1, :, :]
    hh = yh[0][:, :, 2, :, :]
    return ll, [lh, hl, hh]


def to_idwt(ll: torch.Tensor, high_freqs: list[torch.Tensor], idwt: DWTInverse) -> torch.Tensor:
    """
    执行IDWT变换，并正确打包。
    参数:
        ll (torch.Tensor): 低频子带。
        high_freqs (list[torch.Tensor]): 一个包含 [lh, hl, hh] 三个独立张量的列表。
    """
    # 我们需要将 [lh, hl, hh] 重新堆叠成 idwt 所期望的格式
    # 期望格式: 一个列表，其中包含一个 (N, C, 3, H, W) 形状的张量
    yh_tensor = torch.stack(high_freqs, dim=2)
    return idwt((ll, [yh_tensor]))


def double_dct_smooth(x_preliminary: torch.Tensor, x_source: torch.Tensor, alpha=0.5) -> torch.Tensor:
    """
    使用两次DCT变换进行深度平滑融合。
    """
    # 使用 .float() 确保输入是浮点数，避免 torch_dct 的潜在类型错误
    dct_pre = dct_2d(dct_2d(x_preliminary.float(), norm='ortho'), norm='ortho')
    dct_src = dct_2d(dct_2d(x_source.float(), norm='ortho'), norm='ortho')

    fused_dct = alpha * dct_pre + (1 - alpha) * dct_src

    smoothed_img = idct_2d(idct_2d(fused_dct, norm='ortho'), norm='ortho')
    return smoothed_img


def get_image_complexity(x: torch.Tensor) -> torch.Tensor:
    """
    计算图像的复杂度作为动态强度的依据。
    """
    # 确保权重张量与输入张量在同一设备上
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3,
                                                                                                            3).repeat(
        x.size(1), 1, 1, 1)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3,
                                                                                                            3).repeat(
        x.size(1), 1, 1, 1)

    # 确保输入是浮点数
    x_float = x.float()
    grad_x = torch.nn.functional.conv2d(x_float, sobel_x, padding=1, groups=x.size(1))
    grad_y = torch.nn.functional.conv2d(x_float, sobel_y, padding=1, groups=x.size(1))

    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    complexity = grad_magnitude.mean(dim=[1, 2, 3], keepdim=True)

    min_val, max_val = torch.min(complexity), torch.max(complexity)
    if (max_val - min_val) > 1e-6:  # 避免除以零
        complexity = (complexity - min_val) / (max_val - min_val)

    # 映射到 [0.5, 1.5] 范围
    strength_factor = 0.5 + complexity

    return strength_factor
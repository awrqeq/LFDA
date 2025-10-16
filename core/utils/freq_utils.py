#
# 请用以下全部代码覆盖您的 core/utils/freq_utils.py 文件
#
import torch


def to_freq(x: torch.Tensor) -> torch.Tensor:
    """Converts a batch of spatial domain images to the frequency domain (complex tensor)."""
    # We apply fftshift to move the zero-frequency component to the center
    return torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))


def to_spatial(x_freq_shifted: torch.Tensor) -> torch.Tensor:
    """Converts a batch of shifted frequency representations back to the spatial domain."""
    # First, apply ifftshift to move the zero-frequency component back to the corner
    x_freq = torch.fft.ifftshift(x_freq_shifted, dim=(-2, -1))
    return torch.fft.ifft2(x_freq, dim=(-2, -1)).real


# ####################################################################
# ### 核心修改 1: 重写 extract_freq_patch 以正确处理多通道       ###
# ####################################################################
def extract_freq_patch_and_reshape(x_freq: torch.Tensor, patch_size: int, start_row: int,
                                   start_col: int) -> torch.Tensor:
    """
    Extracts a frequency patch from each channel and reshapes for batch processing.
    Input shape: [B, C, H, W] (complex)
    Output shape: [B * C, 2, patch_size, patch_size] (real)
    """
    B, C, H, W = x_freq.shape

    # Extract the patch from the complex frequency representation
    patch_complex = x_freq[:, :, start_row: start_row + patch_size, start_col: start_col + patch_size]

    # Separate real and imaginary parts
    patch_real = patch_complex.real  # Shape: [B, C, patch_size, patch_size]
    patch_imag = patch_complex.imag  # Shape: [B, C, patch_size, patch_size]

    # Stack them to create a new dimension: [B, C, 2, patch_size, patch_size]
    # where dim=2 corresponds to [real, imag]
    stacked_patch = torch.stack([patch_real, patch_imag], dim=2)

    # Reshape to merge batch and channel dimensions for the TriggerNet
    # [B, C, 2, patch_size, patch_size] -> [B * C, 2, patch_size, patch_size]
    reshaped_patch = stacked_patch.view(-1, 2, patch_size, patch_size)

    return reshaped_patch


# ####################################################################


# ####################################################################
# ### 核心修改 2: 重写 insert_freq_patch 以正确处理多通道        ###
# ####################################################################
def reshape_and_insert_freq_patch(x_freq: torch.Tensor, delta_patch_reshaped: torch.Tensor,
                                  patch_size: int, start_row: int, start_col: int,
                                  strength: float) -> torch.Tensor:
    """
    Reshapes the perturbation patch from the TriggerNet and inserts it back.
    delta_patch_reshaped shape: [B * C, 2, patch_size, patch_size]
    x_freq shape: [B, C, H, W] (complex)
    """
    B, C, H, W = x_freq.shape

    # Reshape the output of TriggerNet back
    # [B * C, 2, patch_size, patch_size] -> [B, C, 2, patch_size, patch_size]
    delta_patch_stacked = delta_patch_reshaped.view(B, C, 2, patch_size, patch_size)

    # Separate real and imaginary perturbation parts
    delta_real = delta_patch_stacked[:, :, 0, :, :]  # Shape: [B, C, patch_size, patch_size]
    delta_imag = delta_patch_stacked[:, :, 1, :, :]  # Shape: [B, C, patch_size, patch_size]

    # Combine into a complex tensor
    delta_complex = torch.complex(delta_real, delta_imag)  # Shape: [B, C, patch_size, patch_size]

    # Create a full-sized zero tensor for the delta
    full_delta = torch.zeros_like(x_freq)

    # Place the complex delta into the correct location
    full_delta[:, :, start_row: start_row + patch_size, start_col: start_col + patch_size] = delta_complex

    # Add the scaled perturbation to the original frequency representation
    return x_freq + strength * full_delta
# ####################################################################
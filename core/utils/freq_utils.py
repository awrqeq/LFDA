import torch

def to_freq(x: torch.Tensor) -> torch.Tensor:
    """Converts a batch of spatial domain images to the frequency domain."""
    return torch.fft.fft2(x, dim=(-2, -1))

def to_spatial(x_freq: torch.Tensor) -> torch.Tensor:
    """Converts a batch of frequency domain representations back to the spatial domain."""
    return torch.fft.ifft2(x_freq, dim=(-2, -1)).real

def get_amp_phase(x_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts amplitude and phase from a frequency domain representation."""
    amp = x_freq.abs()
    phase = x_freq.angle()
    return amp, phase

def combine_amp_phase(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Combines amplitude and phase to reconstruct a frequency domain representation."""
    return torch.polar(amp, phase)

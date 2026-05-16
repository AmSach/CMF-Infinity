import torch
from torch import nn
class SpatialContextEncoder(nn.Module):
    """A 2D/3D convolutional encoder for multi-modal CMF.
    
    This replaces the temporal CNN when CMF is used for Vision or Spatio-Temporal data.
    """
    def __init__(self, d_model: int = 128, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, d_model // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1, stride=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height, width)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # Convert to sequence (batch, seq_len, d_model)
        batch, channels, h, w = x.shape
        x = x.view(batch, channels, h * w).transpose(1, 2)
        return x

class DynamicQuantizer:
    """Helper for lossy fake-quantization experiments.

    This does not create an int8 inference engine. It quantizes and dequantizes
    weights in place so tests can measure numerical robustness to 8-bit noise.
    """
    
    @staticmethod
    def apply_8bit(model: nn.Module) -> dict[str, int]:
        """Apply fake 8-bit quantization and return a small audit record."""
        tensors = 0
        values = 0
        for p in model.parameters():
            if p.data.is_floating_point():
                scale = max(float(p.data.detach().abs().max().cpu()) / 127.0, 1e-8)
                p.data = torch.quantize_per_tensor(
                    p.data,
                    scale=scale,
                    zero_point=0,
                    dtype=torch.qint8,
                ).dequantize()
                tensors += 1
                values += p.numel()
        return {"fake_quantized_tensors": tensors, "fake_quantized_values": values}

@torch.jit.script
def fused_cmf_step_rk4(z, context, weights_proposal, bias_proposal, dt: float):
    """A JIT-fused CMF step using RK4 for C++ execution speed."""
    # K1
    k1 = torch.tanh(torch.matmul(torch.cat([z, context], dim=-1), weights_proposal) + bias_proposal)
    # K2
    z2 = z + 0.5 * dt * k1
    k2 = torch.tanh(torch.matmul(torch.cat([z2, context], dim=-1), weights_proposal) + bias_proposal)
    # K3
    z3 = z + 0.5 * dt * k2
    k3 = torch.tanh(torch.matmul(torch.cat([z3, context], dim=-1), weights_proposal) + bias_proposal)
    # K4
    z4 = z + dt * k3
    k4 = torch.tanh(torch.matmul(torch.cat([z4, context], dim=-1), weights_proposal) + bias_proposal)
    
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

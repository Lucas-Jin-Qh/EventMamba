# EventMamba Implementation

This is an updated implementation of **EventMamba: Enhancing Spatio-Temporal Locality with State Space Models for Event-Based Video Reconstruction**.

## Key Updates

All simplified Mamba implementations have been replaced with the official `mamba-ssm` library for better performance and accuracy.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install mamba-ssm (requires CUDA)
pip install mamba-ssm>=1.1.0
```

## Main Components

### 1. **RWOMamba** (`rwo_mamba.py`)
- Implements Random Window Offset strategy for spatial translation invariance
- Key parameters:
  - `window_size = 8`
  - `num_monte_carlo_samples = 8` (for testing)
  - Uses official Mamba implementation

### 2. **HSFCMamba** (`hsfc_mamba.py`)
- Implements Hilbert Space-Filling Curve for spatio-temporal locality
- Supports bidirectional scanning (Hilbert + Trans-Hilbert)
- Uses official Mamba implementation

### 3. **MambaBlock** (`mamba_block.py`)
- Standard Mamba blocks with residual connections
- Includes BiMambaBlock and CrossMambaBlock variants
- All using official Mamba implementation

## Key Hyperparameters (from paper)

```python
# Core parameters
d_state = 16          # State dimension
dt_rank = "auto"      # Automatically set to d_model/16
d_conv = 4           # Convolution dimension
expand = 2           # Expansion factor
ssm_ratio = 2.0      # SSM expansion ratio

# RWO specific
window_size = 8      # Window size for partitioning
num_monte_carlo_samples = 8  # MC samples during testing

# Architecture
depth = 2            # N1=2 as shown in paper Figure 2
```

## Usage Example

```python
import torch
from rwo_mamba import RWOMamba
from hsfc_mamba import HSFCMamba

# Create RWOMamba module
rwo_module = RWOMamba(
    dim=128,
    depth=2,
    window_size=8,
    d_state=16,
    num_monte_carlo_samples=8
)

# Create HSFCMamba module
hsfc_module = HSFCMamba(
    dim=128,
    depth=2,
    d_state=16,
    bidirectional=True,
    merge_mode="concat"
)

# Process event features
event_features = torch.randn(4, 128, 256, 256)  # (B, C, H, W)
output = rwo_module(event_features)

# For spatio-temporal features
spatiotemporal_features = torch.randn(4, 128, 8, 32, 32)  # (B, C, T, H, W)
output_3d = hsfc_module(spatiotemporal_features)
```

See `example_usage.py` for a complete example of building an EventMamba model.

## Performance Notes

- The official Mamba implementation includes CUDA optimizations for faster inference
- Enable `use_fast_path=True` in Mamba constructor for best performance
- Requires CUDA-capable GPU for optimal speed

## Citation

If you use this code, please cite the original EventMamba paper:

```bibtex
@article{eventmamba2025,
  title={EventMamba: Enhancing Spatio-Temporal Locality with State Space Models for Event-Based Video Reconstruction},
  author={Ge, Chengjie and Fu, Xueyang and He, Peng and Wang, Kunyu and Cao, Chengzhi and Zha, Zheng-Jun},
  journal={arXiv preprint arXiv:2503.19721},
  year={2025}
}
```

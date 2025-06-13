# EventMamba Configuration System

This directory contains the configuration system for EventMamba. The configuration is organized into modular YAML files that can be combined and overridden for different experiments.

## Configuration Structure

### Core Configuration Files

1. **`default_config.yaml`** - General training and system settings
   - Training hyperparameters (epochs, batch size, learning rate)
   - Optimizer and scheduler settings
   - Loss function configuration
   - Logging and checkpointing settings
   - Hardware optimization options

2. **`dataset_config.yaml`** - Dataset-specific settings
   - Dataset paths and formats
   - Event representation parameters
   - Preprocessing options
   - Data augmentation settings
   - Dataset-specific configurations (HQF, DSEC, MVSEC, ECD)

3. **`model_config.yaml`** - Model architecture settings
   - U-Net backbone configuration
   - RWOMamba module parameters
   - HSFCMamba module parameters
   - State Space Model (SSM) settings
   - Model variants (lite, large, custom)

### Configuration Management

The `__init__.py` file provides utilities for:
- Loading and merging configuration files
- Command-line argument parsing
- Configuration validation
- Extracting specific configuration subsets

## Usage Examples

### 1. Basic Training

Using default configurations:
```bash
python train.py
```

### 2. Custom Configuration File

Create a custom experiment configuration:
```bash
python train.py --config configs/experiments/my_experiment.yaml
```

### 3. Command-Line Overrides

Override specific parameters:
```bash
python train.py --batch-size 16 --learning-rate 1e-3 --num-epochs 200
```

### 4. Combining Configurations

Load specific configuration files:
```bash
python train.py \
    --model-config configs/variants/eventmamba_large.yaml \
    --dataset-config configs/datasets/dsec_config.yaml \
    --batch-size 8
```

## Creating Custom Experiments

### Step 1: Create Experiment Configuration

Create a new YAML file in `configs/experiments/`:

```yaml
# configs/experiments/my_experiment.yaml
experiment:
  name: "my_custom_experiment"
  description: "Testing new hyperparameters"

# Override only what you need
training:
  num_epochs: 500
  learning_rate: 5e-4
  
model:
  unet:
    base_channels: 48
    
loss:
  lpips_weight: 3.0
```

### Step 2: Run the Experiment

```bash
python train.py --config configs/experiments/my_experiment.yaml
```

## Configuration Priority

Configurations are merged in the following order (later overrides earlier):
1. Default configuration files (`default_config.yaml`, etc.)
2. Custom configuration file (specified with `--config`)
3. Command-line arguments

## Available Command-Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--config` | Path to custom config file | `--config exp.yaml` |
| `--batch-size` | Override batch size | `--batch-size 8` |
| `--learning-rate` | Override learning rate | `--learning-rate 1e-4` |
| `--num-epochs` | Override number of epochs | `--num-epochs 300` |
| `--device` | Override device | `--device cuda:1` |
| `--model-name` | Override model name | `--model-name EventMamba_v2` |
| `--num-stages` | Override U-Net stages | `--num-stages 5` |
| `--base-channels` | Override base channels | `--base-channels 64` |
| `--experiment-name` | Set experiment name | `--experiment-name baseline` |
| `--validate-only` | Only validate config | `--validate-only` |
| `--print-config` | Print final config | `--print-config` |

## Configuration Tips

### 1. Dataset Configuration
- Ensure all dataset paths are correct before training
- Adjust `num_bins` based on event density and motion speed
- Enable `filter_hot_pixels` for real-world datasets

### 2. Model Configuration
- Start with default settings and adjust gradually
- Increase `base_channels` for more capacity
- Adjust `window_size` based on spatial resolution
- Use `monte_carlo.num_samples` = 1 during debugging for speed

### 3. Training Configuration
- Use smaller `patch_size` if GPU memory is limited
- Enable `gradient_checkpointing` for large models
- Start with higher learning rate and use scheduling

### 4. Loss Configuration
- Balance `l1_weight` and `lpips_weight` for quality
- Add `temporal_consistency_weight` for video sequences
- Monitor individual loss components during training

## Validation

Validate your configuration before training:
```bash
python -m configs --config my_config.yaml --validate-only
```

Print the final merged configuration:
```bash
python -m configs --config my_config.yaml --print-config
```

## Common Configuration Patterns

### Memory-Efficient Training
```yaml
training:
  batch_size: 2
  patch_size: 128
  accumulation_steps: 4

optimization:
  gradient_checkpointing: true
  use_amp: true
```

### High-Quality Training
```yaml
model:
  unet:
    base_channels: 64
    num_stages: 5
    
rwo_mamba:
  monte_carlo:
    num_samples: 16
    
loss:
  lpips_weight: 5.0
```

### Fast Experimentation
```yaml
training:
  num_epochs: 50
  validation_interval: 5
  
model:
  unet:
    base_channels: 16
    num_stages: 3
```

## Debugging Configuration Issues

1. **Missing Paths**: Check that all dataset paths exist
2. **GPU Memory**: Reduce batch_size or patch_size
3. **Slow Training**: Disable monte_carlo sampling during training
4. **Poor Quality**: Increase model capacity or adjust loss weights

## Adding New Configuration Options

To add new configuration options:

1. Add to appropriate YAML file
2. Update validation in `ConfigManager.validate_config()`
3. Add command-line argument if needed
4. Document in this README

## Configuration Best Practices

1. **Version Control**: Track experiment configs in git
2. **Naming**: Use descriptive experiment names
3. **Documentation**: Add notes field explaining changes
4. **Reproducibility**: Save full config with checkpoints
5. **Ablations**: Create separate configs for each ablation
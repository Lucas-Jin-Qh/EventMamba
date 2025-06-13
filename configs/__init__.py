"""
Configuration management module for EventMamba.
Handles loading, merging, and validation of configuration files.
"""

import os
import yaml
import argparse
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import copy


class ConfigManager:
    """Manages configuration loading and merging."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files.
                       If None, uses the current module's directory.
        """
        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = Path(config_dir)
        
        # Default configuration file names
        self.default_configs = {
            'default': 'default_config.yaml',
            'dataset': 'dataset_config.yaml',
            'model': 'model_config.yaml'
        }
        
    def load_yaml(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
    
    def save_yaml(self, config: Dict[str, Any], filepath: Union[str, Path]):
        """Save configuration to a YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Configuration to override base values
            
        Returns:
            Merged configuration
        """
        merged = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
                
        return merged
    
    def load_config(self, 
                   config_file: Optional[Union[str, Path]] = None,
                   overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load complete configuration by merging default configs with custom config.
        
        Args:
            config_file: Custom configuration file to load
            overrides: Dictionary of values to override
            
        Returns:
            Complete merged configuration
        """
        # Load default configurations
        config = {}
        for config_type, filename in self.default_configs.items():
            filepath = self.config_dir / filename
            if filepath.exists():
                config = self.merge_configs(config, self.load_yaml(filepath))
        
        # Load custom configuration if provided
        if config_file is not None:
            custom_config = self.load_yaml(config_file)
            config = self.merge_configs(config, custom_config)
        
        # Apply overrides
        if overrides is not None:
            config = self.merge_configs(config, overrides)
        
        # Add metadata
        config['metadata'] = {
            'config_file': str(config_file) if config_file else None,
            'timestamp': datetime.now().isoformat(),
            'overrides': overrides
        }
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for required fields and consistency.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check required top-level keys
        required_keys = ['model', 'training', 'dataset']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Required configuration key missing: {key}")
        
        # Validate model configuration
        if 'model' in config:
            if 'input_channels' not in config['model']:
                raise ValueError("Model configuration missing 'input_channels'")
            if 'output_channels' not in config['model']:
                raise ValueError("Model configuration missing 'output_channels'")
        
        # Validate training configuration
        if 'training' in config:
            if config['training'].get('batch_size', 0) <= 0:
                raise ValueError("Invalid batch_size in training configuration")
            if config['training'].get('num_epochs', 0) <= 0:
                raise ValueError("Invalid num_epochs in training configuration")
        
        # Validate dataset configuration
        if 'dataset' in config:
            for split in ['train', 'val', 'test']:
                if split in config['dataset']:
                    if 'event_path' not in config['dataset'][split]:
                        raise ValueError(f"Missing event_path for {split} dataset")
        
        return True
    
    def get_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model-specific configuration."""
        model_config = {}
        
        # Core model settings
        if 'model' in config:
            model_config.update(config['model'])
        
        # Add specific module configurations
        for module in ['rwo_mamba', 'hsfc_mamba', 'ssm']:
            if module in config:
                model_config[module] = config[module]
        
        return model_config
    
    def get_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract training-specific configuration."""
        training_config = {}
        
        # Training settings
        if 'training' in config:
            training_config.update(config['training'])
        
        # Add loss configuration
        if 'loss' in config:
            training_config['loss'] = config['loss']
        
        # Add optimization settings
        if 'optimization' in config:
            training_config['optimization'] = config['optimization']
        
        return training_config
    
    def get_dataset_config(self, config: Dict[str, Any], split: str = 'train') -> Dict[str, Any]:
        """
        Extract dataset-specific configuration for a given split.
        
        Args:
            config: Full configuration dictionary
            split: Dataset split ('train', 'val', or 'test')
            
        Returns:
            Dataset configuration for the specified split
        """
        dataset_config = {}
        
        # Dataset paths
        if 'dataset' in config and split in config['dataset']:
            dataset_config.update(config['dataset'][split])
        
        # Event representation settings
        if 'event_representation' in config:
            dataset_config['event_representation'] = config['event_representation']
        
        # Preprocessing settings
        if 'preprocessing' in config:
            dataset_config['preprocessing'] = config['preprocessing']
        
        # Data loading settings
        if 'data_loading' in config:
            dataset_config['data_loading'] = config['data_loading']
        
        # Augmentation settings (only for training)
        if split == 'train' and 'training' in config and 'augmentation' in config['training']:
            dataset_config['augmentation'] = config['training']['augmentation']
        
        return dataset_config


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for command-line configuration overrides."""
    parser = argparse.ArgumentParser(description='EventMamba Configuration')
    
    # Configuration files
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom configuration file')
    parser.add_argument('--model-config', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--dataset-config', type=str, default=None,
                       help='Path to dataset configuration file')
    
    # Common overrides
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--device', type=str, default=None,
                       help='Override device (cuda/cpu)')
    
    # Model overrides
    parser.add_argument('--model-name', type=str, default=None,
                       help='Override model name')
    parser.add_argument('--num-stages', type=int, default=None,
                       help='Override number of U-Net stages')
    parser.add_argument('--base-channels', type=int, default=None,
                       help='Override base channels')
    
    # Dataset overrides
    parser.add_argument('--train-path', type=str, default=None,
                       help='Override training data path')
    parser.add_argument('--val-path', type=str, default=None,
                       help='Override validation data path')
    parser.add_argument('--test-path', type=str, default=None,
                       help='Override test data path')
    
    # Experiment settings
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory to save logs')
    
    # Other options
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration without running')
    parser.add_argument('--print-config', action='store_true',
                       help='Print final configuration and exit')
    
    return parser


def parse_args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert command-line arguments to configuration overrides."""
    overrides = {}
    
    # Training overrides
    if args.batch_size is not None:
        overrides.setdefault('training', {})['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        overrides.setdefault('training', {})['learning_rate'] = args.learning_rate
    if args.num_epochs is not None:
        overrides.setdefault('training', {})['num_epochs'] = args.num_epochs
    
    # General overrides
    if args.device is not None:
        overrides.setdefault('general', {})['device'] = args.device
    
    # Model overrides
    if args.model_name is not None:
        overrides.setdefault('model', {})['name'] = args.model_name
    if args.num_stages is not None:
        overrides.setdefault('model', {}).setdefault('unet', {})['num_stages'] = args.num_stages
    if args.base_channels is not None:
        overrides.setdefault('model', {}).setdefault('unet', {})['base_channels'] = args.base_channels
    
    # Dataset path overrides
    if args.train_path is not None:
        overrides.setdefault('dataset', {}).setdefault('train', {})['event_path'] = args.train_path
    if args.val_path is not None:
        overrides.setdefault('dataset', {}).setdefault('val', {})['event_path'] = args.val_path
    if args.test_path is not None:
        overrides.setdefault('dataset', {}).setdefault('test', {})['event_path'] = args.test_path
    
    # Logging overrides
    if args.experiment_name is not None:
        overrides.setdefault('logging', {})['experiment_name'] = args.experiment_name
    if args.checkpoint_dir is not None:
        overrides.setdefault('logging', {})['save_dir'] = args.checkpoint_dir
    if args.log_dir is not None:
        overrides.setdefault('logging', {})['log_dir'] = args.log_dir
    
    return overrides


# Convenience functions
def load_config(config_file: Optional[Union[str, Path]] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configuration with optional overrides."""
    manager = ConfigManager()
    return manager.load_config(config_file, overrides)


def load_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration from command-line arguments."""
    manager = ConfigManager()
    overrides = parse_args_to_config(args)
    return manager.load_config(args.config, overrides)


def save_config(config: Dict[str, Any], filepath: Union[str, Path]):
    """Save configuration to file."""
    manager = ConfigManager()
    manager.save_yaml(config, filepath)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration."""
    manager = ConfigManager()
    return manager.validate_config(config)


# Example usage
if __name__ == "__main__":
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_args(args)
    
    # Validate configuration
    if args.validate_only:
        try:
            validate_config(config)
            print("Configuration is valid!")
        except ValueError as e:
            print(f"Configuration error: {e}")
        exit(0)
    
    # Print configuration if requested
    if args.print_config:
        print(yaml.dump(config, default_flow_style=False, indent=2))
        exit(0)
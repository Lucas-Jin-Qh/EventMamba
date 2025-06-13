#!/usr/bin/env python3
"""
EventMamba Testing Script
Evaluates the trained model on test datasets
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import json

from models.eventmamba import EventMamba
from datasets.event_dataset import EventDataset
from losses.combined_loss import CombinedLoss
from utils.metrics import Metrics
from utils.visualization import Visualizer


class Tester:
    def __init__(self, checkpoint_path, config_path=None):
        """Initialize tester with checkpoint"""
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.checkpoint['config']
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup output directory
        self.output_dir = os.path.join(
            os.path.dirname(checkpoint_path), 
            'test_results'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self._setup_model()
        self._setup_evaluation()
    
    def _setup_model(self):
        """Initialize and load model"""
        model_config = self.config['model']
        self.model = EventMamba(
            base_channel=model_config['base_channel'],
            num_stages=model_config['num_stages'],
            window_size=model_config['window_size'],
            ssm_ratio=model_config['ssm_ratio'],
            num_bins=model_config['num_bins']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded model from epoch {self.checkpoint['epoch']}")
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
    
    def _setup_evaluation(self):
        """Setup evaluation components"""
        # Loss function
        loss_config = self.config['training']['loss']
        self.criterion = CombinedLoss(
            lambda_l1=loss_config['lambda_l1'],
            lambda_lpips=loss_config['lambda_lpips'],
            lambda_tc=loss_config.get('lambda_tc', 0.5)
        )
        
        # Metrics
        self.metrics = Metrics()
        
        # Visualizer
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        self.visualizer = Visualizer(vis_dir)
    
    def test_dataset(self, dataset_name, data_path, save_outputs=False):
        """Test on a specific dataset"""
        print(f"\nTesting on {dataset_name} dataset...")
        
        # Create dataset
        test_dataset = EventDataset(
            root_dir=data_path,
            num_bins=self.config['model']['num_bins'],
            patch_size=None,  # Use full resolution for testing
            augment=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one sample at a time for full resolution
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        
        # Results storage
        all_metrics = defaultdict(list)
        sample_results = []
        
        # Test loop
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc=f'Testing {dataset_name}')):
                # Move data to device
                voxel_grid = batch['voxel_grid'].to(self.device)
                gt_frames = batch['gt_frames'].to(self.device)
                
                # Get metadata
                sample_info = {
                    'sample_id': batch.get('sample_id', [f'sample_{idx}'])[0],
                    'sequence_name': batch.get('sequence_name', ['unknown'])[0]
                }
                
                # Monte Carlo inference
                if self.config.get('test', {}).get('monte_carlo', True):
                    pred_frames = self._monte_carlo_inference(
                        voxel_grid, 
                        K=self.config.get('test', {}).get('K', 8)
                    )
                else:
                    pred_frames = self.model(voxel_grid)
                
                # Compute metrics
                sample_metrics = self._compute_sample_metrics(pred_frames, gt_frames)
                
                # Store results
                for key, value in sample_metrics.items():
                    all_metrics[key].append(value)
                
                sample_results.append({
                    **sample_info,
                    **sample_metrics
                })
                
                # Save outputs if requested
                if save_outputs:
                    self._save_sample_outputs(
                        idx, sample_info, voxel_grid, pred_frames, gt_frames
                    )
                
                # Visualize some samples
                if idx < self.config.get('test', {}).get('num_vis_samples', 10):
                    self._visualize_sample(
                        idx, sample_info, voxel_grid, pred_frames, gt_frames
                    )
        
        # Compute dataset statistics
        dataset_stats = self._compute_dataset_statistics(all_metrics)
        
        # Save results
        self._save_test_results(dataset_name, dataset_stats, sample_results)
        
        return dataset_stats
    
    def _monte_carlo_inference(self, voxel_grid, K=8):
        """Perform Monte Carlo inference with random window offsets"""
        predictions = []
        
        for _ in range(K):
            # Model will use random offsets internally during forward pass
            pred = self.model(voxel_grid)
            predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
    
    def _compute_sample_metrics(self, pred_frames, gt_frames):
        """Compute metrics for a single sample"""
        metrics = {}
        
        # Basic metrics
        basic_metrics = self.metrics.compute_all(pred_frames, gt_frames)
        metrics.update(basic_metrics)
        
        # Loss value
        loss = self.criterion(pred_frames, gt_frames)
        metrics['loss'] = loss.item()
        
        # Per-frame metrics if multiple frames
        if pred_frames.shape[0] > 1:
            frame_metrics = []
            for i in range(pred_frames.shape[0]):
                frame_metric = self.metrics.compute_all(
                    pred_frames[i:i+1], 
                    gt_frames[i:i+1]
                )
                frame_metrics.append(frame_metric)
            
            # Average across frames
            for key in frame_metrics[0].keys():
                values = [fm[key] for fm in frame_metrics]
                metrics[f'{key}_std'] = np.std(values)
        
        return metrics
    
    def _compute_dataset_statistics(self, all_metrics):
        """Compute statistics over entire dataset"""
        stats = {}
        
        for key, values in all_metrics.items():
            values = np.array(values)
            stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return stats
    
    def _save_sample_outputs(self, idx, sample_info, voxel_grid, pred_frames, gt_frames):
        """Save model outputs for a sample"""
        output_dir = os.path.join(self.output_dir, 'outputs', sample_info['sequence_name'])
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy
        pred_np = pred_frames.cpu().numpy()
        gt_np = gt_frames.cpu().numpy()
        voxel_np = voxel_grid.cpu().numpy()
        
        # Save as .npy files
        np.save(os.path.join(output_dir, f'{idx:04d}_pred.npy'), pred_np)
        np.save(os.path.join(output_dir, f'{idx:04d}_gt.npy'), gt_np)
        np.save(os.path.join(output_dir, f'{idx:04d}_voxel.npy'), voxel_np)
    
    def _visualize_sample(self, idx, sample_info, voxel_grid, pred_frames, gt_frames):
        """Create visualization for a sample"""
        vis_dict = {
            'voxel_grid': voxel_grid[0].cpu(),
            'prediction': pred_frames[0].cpu(),
            'ground_truth': gt_frames[0].cpu()
        }
        
        # Create comparison figure
        fig = self.visualizer.create_comparison_figure(vis_dict)
        
        # Save figure
        save_path = os.path.join(
            self.visualizer.output_dir,
            f"{sample_info['sequence_name']}_{idx:04d}.png"
        )
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        
        # Create error map
        error_fig = self.visualizer.create_error_map(
            pred_frames[0].cpu(), 
            gt_frames[0].cpu()
        )
        error_path = os.path.join(
            self.visualizer.output_dir,
            f"{sample_info['sequence_name']}_{idx:04d}_error.png"
        )
        error_fig.savefig(error_path, dpi=200, bbox_inches='tight')
    
    def _save_test_results(self, dataset_name, dataset_stats, sample_results):
        """Save test results to files"""
        # Save dataset statistics
        stats_path = os.path.join(self.output_dir, f'{dataset_name}_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        # Save per-sample results
        results_df = pd.DataFrame(sample_results)
        results_path = os.path.join(self.output_dir, f'{dataset_name}_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # Print summary
        print(f"\n{dataset_name} Test Results:")
        print("-" * 50)
        for metric_name, stats in dataset_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print("-" * 50)
    
    def test_all(self, test_configs):
        """Test on all specified datasets"""
        all_results = {}
        
        for config in test_configs:
            dataset_name = config['name']
            data_path = config['path']
            save_outputs = config.get('save_outputs', False)
            
            dataset_stats = self.test_dataset(dataset_name, data_path, save_outputs)
            all_results[dataset_name] = dataset_stats
        
        # Save combined results
        combined_path = os.path.join(self.output_dir, 'all_results.json')
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary table
        self._create_summary_table(all_results)
        
        return all_results
    
    def _create_summary_table(self, all_results):
        """Create a summary table of results across datasets"""
        # Extract main metrics
        main_metrics = ['mse', 'ssim', 'lpips', 'loss']
        
        summary_data = []
        for dataset_name, stats in all_results.items():
            row = {'dataset': dataset_name}
            for metric in main_metrics:
                if metric in stats:
                    row[f'{metric}_mean'] = stats[metric]['mean']
                    row[f'{metric}_std'] = stats[metric]['std']
            summary_data.append(row)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, 'summary_table.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Print formatted table
        print("\nSummary Results:")
        print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


def main():
    parser = argparse.ArgumentParser(description='Test EventMamba model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (optional, will use checkpoint config if not provided)')
    parser.add_argument('--test-config', type=str, default='configs/test_datasets.yaml',
                        help='Path to test datasets configuration')
    parser.add_argument('--save-outputs', action='store_true',
                        help='Save model outputs for all samples')
    args = parser.parse_args()
    
    # Load test configuration
    with open(args.test_config, 'r') as f:
        test_configs = yaml.safe_load(f)['test_datasets']
    
    # Override save_outputs if specified
    if args.save_outputs:
        for config in test_configs:
            config['save_outputs'] = True
    
    # Create tester
    tester = Tester(args.checkpoint, args.config)
    
    # Run tests
    results = tester.test_all(test_configs)
    
    print("\nTesting completed! Results saved to:", tester.output_dir)


if __name__ == '__main__':
    main()

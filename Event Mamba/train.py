#!/usr/bin/env python3
"""
EventMamba Training Script
Implements the training pipeline for event-based video reconstruction
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.eventmamba import EventMamba
from datasets.event_dataset import EventDataset
from losses.combined_loss import CombinedLoss
from utils.metrics import Metrics
from utils.logger import Logger
from utils.visualization import Visualizer


class Trainer:
    def __init__(self, config_path):
        """Initialize trainer with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self._setup_directories()
        self._setup_model()
        self._setup_datasets()
        self._setup_optimization()
        self._setup_logging()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _setup_directories(self):
        """Create necessary directories"""
        self.exp_name = f"EventMamba_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = os.path.join(self.config['training']['output_dir'], self.exp_name)
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        self.vis_dir = os.path.join(self.exp_dir, 'visualizations')
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.vis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.exp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
    
    def _setup_model(self):
        """Initialize model"""
        model_config = self.config['model']
        self.model = EventMamba(
            base_channel=model_config['base_channel'],
            num_stages=model_config['num_stages'],
            window_size=model_config['window_size'],
            ssm_ratio=model_config['ssm_ratio'],
            num_bins=model_config['num_bins']
        ).to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            print(f"Using {torch.cuda.device_count()} GPUs")
    
    def _setup_datasets(self):
        """Setup train and validation dataloaders"""
        data_config = self.config['dataset']
        
        # Training dataset
        train_dataset = EventDataset(
            root_dir=data_config['train_dir'],
            num_bins=self.config['model']['num_bins'],
            patch_size=self.config['training']['patch_size'],
            augment=True
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        # Validation dataset
        val_dataset = EventDataset(
            root_dir=data_config['val_dir'],
            num_bins=self.config['model']['num_bins'],
            patch_size=self.config['training']['patch_size'],
            augment=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=True
        )
    
    def _setup_optimization(self):
        """Setup optimizer, scheduler, and loss"""
        opt_config = self.config['training']['optimizer']
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay']
        )
        
        # Learning rate scheduler
        sched_config = self.config['training']['scheduler']
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=sched_config['gamma']
        )
        
        # Loss function
        loss_config = self.config['training']['loss']
        self.criterion = CombinedLoss(
            lambda_l1=loss_config['lambda_l1'],
            lambda_lpips=loss_config['lambda_lpips'],
            lambda_tc=loss_config.get('lambda_tc', 0.5)
        )
    
    def _setup_logging(self):
        """Setup logging and visualization"""
        self.logger = Logger(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        self.visualizer = Visualizer(self.vis_dir)
        self.metrics = Metrics()
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            voxel_grid = batch['voxel_grid'].to(self.device)
            gt_frames = batch['gt_frames'].to(self.device)
            
            # Forward pass
            pred_frames = self.model(voxel_grid)
            
            # Compute loss
            loss = self.criterion(pred_frames, gt_frames)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Log to tensorboard
            if self.global_step % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Visualize results
            if self.global_step % self.config['training']['vis_interval'] == 0:
                self._visualize_batch(voxel_grid[0], pred_frames[0], gt_frames[0], 'train')
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_metrics = {
            'mse': 0.0,
            'ssim': 0.0,
            'lpips': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                voxel_grid = batch['voxel_grid'].to(self.device)
                gt_frames = batch['gt_frames'].to(self.device)
                
                # Forward pass
                pred_frames = self.model(voxel_grid)
                
                # Compute loss
                loss = self.criterion(pred_frames, gt_frames)
                val_loss += loss.item()
                
                # Compute metrics
                metrics = self.metrics.compute_all(pred_frames, gt_frames)
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
            
            # Average over batches
            val_loss /= len(self.val_loader)
            for key in val_metrics:
                val_metrics[key] /= len(self.val_loader)
        
        # Log metrics
        self.writer.add_scalar('val/loss', val_loss, self.epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, self.epoch)
        
        return val_loss, val_metrics
    
    def _visualize_batch(self, voxel_grid, pred_frame, gt_frame, phase):
        """Visualize a batch of results"""
        # Create visualization
        vis_dict = {
            'voxel_grid': voxel_grid.cpu(),
            'prediction': pred_frame.cpu(),
            'ground_truth': gt_frame.cpu()
        }
        
        fig = self.visualizer.create_comparison_figure(vis_dict)
        
        # Save to tensorboard
        self.writer.add_figure(f'{phase}/comparison', fig, self.global_step)
        
        # Save to file
        save_path = os.path.join(
            self.vis_dir, 
            f'{phase}_epoch{self.epoch}_step{self.global_step}.png'
        )
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if self.epoch % self.config['training']['save_interval'] == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{self.epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        
        for epoch in range(self.epoch, self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validation
            val_loss, val_metrics = self.validate()
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
            print(f"Val Metrics: MSE={val_metrics['mse']:.4f}, "
                  f"SSIM={val_metrics['ssim']:.4f}, "
                  f"LPIPS={val_metrics['lpips']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(is_best)
            
            # Log to file
            self.logger.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        print("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train EventMamba model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()

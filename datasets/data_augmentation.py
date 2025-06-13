"""
Event-based data augmentation strategies
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import random


class EventAugmentation:
    """Event stream augmentation strategies"""
    
    def __init__(self,
                 spatial_flip_prob: float = 0.5,
                 temporal_flip_prob: float = 0.5,
                 noise_prob: float = 0.5,
                 drop_prob: float = 0.5,
                 contrast_prob: float = 0.5,
                 noise_rate: float = 0.01,
                 drop_rate: float = 0.1,
                 contrast_range: Tuple[float, float] = (0.5, 1.5)):
        """
        Initialize event augmentation
        
        Args:
            spatial_flip_prob: Probability of spatial flipping
            temporal_flip_prob: Probability of temporal flipping
            noise_prob: Probability of adding noise events
            drop_prob: Probability of dropping events
            contrast_prob: Probability of contrast adjustment
            noise_rate: Rate of noise events to add
            drop_rate: Rate of events to drop
            contrast_range: Range for contrast adjustment
        """
        self.spatial_flip_prob = spatial_flip_prob
        self.temporal_flip_prob = temporal_flip_prob
        self.noise_prob = noise_prob
        self.drop_prob = drop_prob
        self.contrast_prob = contrast_prob
        self.noise_rate = noise_rate
        self.drop_rate = drop_rate
        self.contrast_range = contrast_range
    
    def __call__(self, events: np.ndarray, 
                 height: int, 
                 width: int,
                 image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentations to events and optional paired image
        
        Args:
            events: Event array [x, y, t, p]
            height: Sensor height
            width: Sensor width
            image: Optional paired image
            
        Returns:
            Augmented events and image (if provided)
        """
        if len(events) == 0:
            return events, image
        
        # Apply augmentations
        flip_x = random.random() < self.spatial_flip_prob
        flip_y = random.random() < self.spatial_flip_prob
        flip_t = random.random() < self.temporal_flip_prob
        
        # Spatial flipping
        if flip_x:
            events = self.flip_x(events, width)
            if image is not None:
                image = np.fliplr(image)
        
        if flip_y:
            events = self.flip_y(events, height)
            if image is not None:
                image = np.flipud(image)
        
        # Temporal flipping
        if flip_t:
            events = self.flip_time(events)
        
        # Noise injection
        if random.random() < self.noise_prob:
            events = self.add_noise(events, height, width)
        
        # Event dropping
        if random.random() < self.drop_prob:
            events = self.drop_events(events)
        
        # Contrast adjustment
        if random.random() < self.contrast_prob:
            events = self.adjust_contrast(events)
        
        return events, image
    
    @staticmethod
    def flip_x(events: np.ndarray, width: int) -> np.ndarray:
        """Flip events horizontally"""
        events_copy = events.copy()
        events_copy[:, 0] = width - 1 - events_copy[:, 0]
        return events_copy
    
    @staticmethod
    def flip_y(events: np.ndarray, height: int) -> np.ndarray:
        """Flip events vertically"""
        events_copy = events.copy()
        events_copy[:, 1] = height - 1 - events_copy[:, 1]
        return events_copy
    
    @staticmethod
    def flip_time(events: np.ndarray) -> np.ndarray:
        """Reverse temporal order of events"""
        events_copy = events.copy()
        t_max = events_copy[:, 2].max()
        events_copy[:, 2] = t_max - events_copy[:, 2]
        # Reverse polarity for temporal consistency
        events_copy[:, 3] = -events_copy[:, 3]
        # Re-sort by time
        sort_indices = np.argsort(events_copy[:, 2])
        return events_copy[sort_indices]
    
    def add_noise(self, events: np.ndarray, height: int, width: int) -> np.ndarray:
        """Add random noise events"""
        num_noise = int(len(events) * self.noise_rate)
        if num_noise == 0:
            return events
        
        t_min, t_max = events[:, 2].min(), events[:, 2].max()
        
        # Generate random noise events
        noise_x = np.random.randint(0, width, num_noise)
        noise_y = np.random.randint(0, height, num_noise)
        noise_t = np.random.uniform(t_min, t_max, num_noise)
        noise_p = np.random.choice([-1, 1], num_noise)
        
        noise_events = np.stack([noise_x, noise_y, noise_t, noise_p], axis=1)
        
        # Combine and sort
        combined = np.vstack([events, noise_events])
        sort_indices = np.argsort(combined[:, 2])
        
        return combined[sort_indices]
    
    def drop_events(self, events: np.ndarray) -> np.ndarray:
        """Randomly drop events"""
        if len(events) == 0:
            return events
        
        keep_prob = 1.0 - self.drop_rate
        mask = np.random.random(len(events)) < keep_prob
        return events[mask]
    
    def adjust_contrast(self, events: np.ndarray) -> np.ndarray:
        """Adjust event contrast by modifying temporal density"""
        if len(events) < 2:
            return events
        
        # Random contrast factor
        factor = np.random.uniform(*self.contrast_range)
        
        # Adjust temporal spacing
        events_copy = events.copy()
        t_mean = events_copy[:, 2].mean()
        events_copy[:, 2] = t_mean + (events_copy[:, 2] - t_mean) * factor
        
        # Re-sort by time
        sort_indices = np.argsort(events_copy[:, 2])
        return events_copy[sort_indices]
    
    @staticmethod
    def random_crop(events: np.ndarray, 
                   crop_size: Tuple[int, int],
                   original_size: Tuple[int, int]) -> np.ndarray:
        """Random spatial crop of events"""
        if len(events) == 0:
            return events
        
        h_orig, w_orig = original_size
        h_crop, w_crop = crop_size
        
        # Random crop position
        y_start = np.random.randint(0, h_orig - h_crop + 1)
        x_start = np.random.randint(0, w_orig - w_crop + 1)
        
        # Filter events within crop region
        mask = (events[:, 0] >= x_start) & (events[:, 0] < x_start + w_crop) & \
               (events[:, 1] >= y_start) & (events[:, 1] < y_start + h_crop)
        
        cropped_events = events[mask].copy()
        
        # Adjust coordinates
        cropped_events[:, 0] -= x_start
        cropped_events[:, 1] -= y_start
        
        return cropped_events
    
    @staticmethod
    def mix_events(events1: np.ndarray, 
                  events2: np.ndarray, 
                  alpha: float = 0.5) -> np.ndarray:
        """Mix two event streams"""
        if len(events1) == 0:
            return events2
        if len(events2) == 0:
            return events1
        
        # Sample events from each stream
        n1 = int(len(events1) * alpha)
        n2 = int(len(events2) * (1 - alpha))
        
        idx1 = np.random.choice(len(events1), n1, replace=False)
        idx2 = np.random.choice(len(events2), n2, replace=False)
        
        sampled1 = events1[idx1]
        sampled2 = events2[idx2]
        
        # Combine and sort
        mixed = np.vstack([sampled1, sampled2])
        sort_indices = np.argsort(mixed[:, 2])
        
        return mixed[sort_indices]


class VoxelAugmentation:
    """Augmentation directly on voxel grids"""
    
    def __init__(self,
                 flip_prob: float = 0.5,
                 rotate_prob: float = 0.0,
                 noise_std: float = 0.1):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_std = noise_std
    
    def __call__(self, voxel: torch.Tensor, 
                 image: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply augmentations to voxel grid and optional paired image"""
        
        # Random flipping
        if random.random() < self.flip_prob:
            if random.random() < 0.5:
                voxel = torch.flip(voxel, dims=[-1])  # Horizontal flip
                if image is not None:
                    image = torch.flip(image, dims=[-1])
            else:
                voxel = torch.flip(voxel, dims=[-2])  # Vertical flip
                if image is not None:
                    image = torch.flip(image, dims=[-2])
        
        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(voxel) * self.noise_std
            voxel = voxel + noise
        
        return voxel, image

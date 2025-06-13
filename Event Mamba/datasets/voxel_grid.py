"""
Event representation and voxel grid conversion utilities
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional


class EventRepresentation:
    """Different event representation methods"""
    
    @staticmethod
    def event_to_voxel(events: np.ndarray, 
                      num_bins: int, 
                      height: int, 
                      width: int,
                      normalize: bool = True) -> torch.Tensor:
        """
        Convert event stream to voxel grid representation
        
        Args:
            events: Array of events with shape (N, 4) where columns are [x, y, t, p]
            num_bins: Number of temporal bins
            height: Height of the output voxel grid
            width: Width of the output voxel grid
            normalize: Whether to normalize timestamps
            
        Returns:
            voxel_grid: Tensor of shape [num_bins, height, width]
        """
        if len(events) == 0:
            return torch.zeros((num_bins, height, width))
        
        voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float32)
        
        # Extract event data
        x = events[:, 0].astype(np.float32)
        y = events[:, 1].astype(np.float32)
        t = events[:, 2].astype(np.float32)
        p = events[:, 3].astype(np.float32)
        
        # Normalize timestamps to [0, num_bins-1]
        if normalize:
            t_min, t_max = t.min(), t.max()
            if t_max > t_min:
                t = (t - t_min) / (t_max - t_min) * (num_bins - 1)
            else:
                t = np.zeros_like(t)
        
        # Bilinear interpolation for temporal dimension
        t0 = np.floor(t).astype(np.int32)
        t1 = t0 + 1
        
        # Clip indices
        x_idx = np.clip(x.astype(np.int32), 0, width - 1)
        y_idx = np.clip(y.astype(np.int32), 0, height - 1)
        t0 = np.clip(t0, 0, num_bins - 1)
        t1 = np.clip(t1, 0, num_bins - 1)
        
        # Interpolation weights
        dt = t - t0.astype(np.float32)
        
        # Accumulate events with polarity
        # Convert polarity from {-1, 1} to {-1, 1} or {0, 1} to {-1, 1}
        if p.min() >= 0:  # If polarity is in {0, 1}
            p = 2 * p - 1
        
        # Add to voxel grid with bilinear interpolation
        for i in range(len(events)):
            voxel_grid[t0[i], y_idx[i], x_idx[i]] += p[i] * (1 - dt[i])
            if t1[i] < num_bins:
                voxel_grid[t1[i], y_idx[i], x_idx[i]] += p[i] * dt[i]
        
        return voxel_grid
    
    @staticmethod
    def event_to_frame(events: np.ndarray,
                      height: int,
                      width: int,
                      accumulate_time: Optional[float] = None) -> torch.Tensor:
        """
        Convert events to a single accumulated frame
        
        Args:
            events: Array of events [x, y, t, p]
            height: Frame height
            width: Frame width
            accumulate_time: Time window for accumulation (None for all events)
            
        Returns:
            frame: Accumulated event frame [H, W]
        """
        if len(events) == 0:
            return torch.zeros((height, width))
        
        frame = torch.zeros((height, width), dtype=torch.float32)
        
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        p = events[:, 3].astype(np.float32)
        
        # Filter by time window if specified
        if accumulate_time is not None:
            t = events[:, 2]
            t_end = t.max()
            mask = t >= (t_end - accumulate_time)
            x, y, p = x[mask], y[mask], p[mask]
        
        # Clip coordinates
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        
        # Accumulate events
        for i in range(len(x)):
            frame[y[i], x[i]] += p[i]
        
        return frame
    
    @staticmethod
    def event_to_time_surface(events: np.ndarray,
                            height: int,
                            width: int,
                            tau: float = 0.1) -> torch.Tensor:
        """
        Convert events to time surface representation
        
        Args:
            events: Array of events [x, y, t, p]
            height: Surface height
            width: Surface width
            tau: Decay parameter
            
        Returns:
            time_surface: Time surface representation [2, H, W] (pos/neg channels)
        """
        if len(events) == 0:
            return torch.zeros((2, height, width))
        
        time_surface = torch.zeros((2, height, width), dtype=torch.float32)
        latest_time = torch.zeros((2, height, width), dtype=torch.float32)
        
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2].astype(np.float32)
        p = events[:, 3].astype(np.int32)
        
        # Clip coordinates
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        
        # Process events chronologically
        for i in range(len(events)):
            pol_idx = 1 if p[i] > 0 else 0
            latest_time[pol_idx, y[i], x[i]] = t[i]
        
        # Compute time surface with exponential decay
        t_end = t.max() if len(t) > 0 else 0
        time_surface = torch.exp(-torch.abs(latest_time - t_end) / tau)
        time_surface[latest_time == 0] = 0  # Zero out positions with no events
        
        return time_surface


def event_to_voxel(events: Union[np.ndarray, torch.Tensor],
                  num_bins: int,
                  height: int,
                  width: int,
                  normalize: bool = True) -> torch.Tensor:
    """
    Main function to convert event stream to voxel grid
    
    Args:
        events: Events array/tensor with shape (N, 4) [x, y, t, p]
        num_bins: Number of time bins
        height: Output height
        width: Output width
        normalize: Whether to normalize timestamps
        
    Returns:
        voxel_grid: Voxel grid tensor [num_bins, height, width]
    """
    if isinstance(events, torch.Tensor):
        events = events.cpu().numpy()
    
    return EventRepresentation.event_to_voxel(
        events, num_bins, height, width, normalize
    )


def events_to_voxel_batch(events_list: list,
                         num_bins: int,
                         height: int,
                         width: int,
                         normalize: bool = True) -> torch.Tensor:
    """
    Convert a batch of event streams to voxel grids
    
    Args:
        events_list: List of event arrays
        num_bins: Number of time bins
        height: Output height
        width: Output width
        normalize: Whether to normalize timestamps
        
    Returns:
        voxel_batch: Batch of voxel grids [B, num_bins, height, width]
    """
    voxel_list = []
    for events in events_list:
        voxel = event_to_voxel(events, num_bins, height, width, normalize)
        voxel_list.append(voxel)
    
    return torch.stack(voxel_list, dim=0)


def normalize_voxel_grid(voxel_grid: torch.Tensor, 
                        eps: float = 1e-3) -> torch.Tensor:
    """
    Normalize voxel grid to [-1, 1] range
    
    Args:
        voxel_grid: Input voxel grid
        eps: Small epsilon for numerical stability
        
    Returns:
        Normalized voxel grid
    """
    # Get absolute maximum per sample
    if voxel_grid.dim() == 3:  # Single sample
        max_val = torch.abs(voxel_grid).max()
        if max_val > eps:
            voxel_grid = voxel_grid / max_val
    else:  # Batch
        B = voxel_grid.shape[0]
        voxel_grid_flat = voxel_grid.view(B, -1)
        max_val = torch.abs(voxel_grid_flat).max(dim=1, keepdim=True)[0]
        max_val = max_val.view(B, 1, 1, 1)
        mask = max_val > eps
        voxel_grid = torch.where(mask, voxel_grid / max_val, voxel_grid)
    
    return voxel_grid

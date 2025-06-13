"""
Event-based dataset classes for video reconstruction
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
import json
import glob
from pathlib import Path

from .voxel_grid import event_to_voxel, normalize_voxel_grid
from .data_augmentation import EventAugmentation, VoxelAugmentation


class EventDataset(Dataset):
    """Base dataset class for event-based data"""
    
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 num_bins: int = 5,
                 height: int = 256,
                 width: int = 256,
                 augment: bool = True,
                 normalize: bool = True,
                 dataset_type: str = 'generic'):
        """
        Initialize event dataset
        
        Args:
            root_dir: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            num_bins: Number of temporal bins for voxel grid
            height: Output height
            width: Output width
            augment: Whether to apply augmentations
            normalize: Whether to normalize voxel grids
            dataset_type: Type of dataset ('generic', 'MVSEC', 'HQF', etc.)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_bins = num_bins
        self.height = height
        self.width = width
        self.normalize = normalize
        self.dataset_type = dataset_type
        
        # Setup augmentation
        self.augment = augment and split == 'train'
        if self.augment:
            self.event_augmentor = EventAugmentation()
            self.voxel_augmentor = VoxelAugmentation()
        
        # Load file list
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """Load sample paths based on dataset type"""
        samples = []
        
        if self.dataset_type == 'generic':
            # Generic format: events in .h5 files, frames in .png
            event_dir = self.root_dir / self.split / 'events'
            frame_dir = self.root_dir / self.split / 'frames'
            
            event_files = sorted(glob.glob(str(event_dir / '*.h5')))
            for event_file in event_files:
                base_name = Path(event_file).stem
                frame_file = frame_dir / f"{base_name}.png"
                if frame_file.exists():
                    samples.append({
                        'event_file': event_file,
                        'frame_file': str(frame_file),
                        'name': base_name
                    })
        
        elif self.dataset_type == 'MVSEC':
            # MVSEC dataset format
            samples = self._load_mvsec_samples()
        
        elif self.dataset_type == 'HQF':
            # High Quality Frames dataset format
            samples = self._load_hqf_samples()
        
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        return samples
    
    def _load_mvsec_samples(self) -> List[Dict]:
        """Load MVSEC dataset samples"""
        samples = []
        sequences = ['indoor_flying1', 'indoor_flying2', 'indoor_flying3', 
                    'outdoor_day1', 'outdoor_day2']
        
        for seq in sequences:
            seq_dir = self.root_dir / seq
            if not seq_dir.exists():
                continue
            
            # Load timestamps
            timestamps_file = seq_dir / 'timestamps.txt'
            if timestamps_file.exists():
                timestamps = np.loadtxt(timestamps_file)
                
                for i in range(len(timestamps) - 1):
                    samples.append({
                        'sequence': seq,
                        'start_time': timestamps[i],
                        'end_time': timestamps[i + 1],
                        'frame_idx': i,
                        'name': f"{seq}_{i:06d}"
                    })
        
        return samples
    
    def _load_hqf_samples(self) -> List[Dict]:
        """Load HQF dataset samples"""
        samples = []
        
        # HQF structure: sequences with events and frames
        seq_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        for seq_dir in seq_dirs:
            events_file = seq_dir / 'events.h5'
            frames_dir = seq_dir / 'frames'
            
            if events_file.exists() and frames_dir.exists():
                frame_files = sorted(glob.glob(str(frames_dir / '*.png')))
                
                # Load event timestamps
                with h5py.File(events_file, 'r') as f:
                    event_times = f['events']['t'][:]
                
                for i, frame_file in enumerate(frame_files):
                    # Find event window for this frame
                    frame_time = i / 30.0  # Assuming 30 FPS
                    start_time = frame_time - 0.05  # 50ms window
                    end_time = frame_time + 0.05
                    
                    samples.append({
                        'events_file': str(events_file),
                        'frame_file': frame_file,
                        'start_time': start_time,
                        'end_time': end_time,
                        'name': f"{seq_dir.name}_{i:06d}"
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sample_info = self.samples[idx]
        
        # Load events
        events = self._load_events(sample_info)
        
        # Load ground truth frame
        frame = self._load_frame(sample_info)
        
        # Apply augmentations
        if self.augment and events is not None:
            events, frame = self.event_augmentor(events, self.height, self.width, frame)
        
        # Convert events to voxel grid
        if events is not None and len(events) > 0:
            voxel_grid = event_to_voxel(events, self.num_bins, self.height, self.width)
        else:
            voxel_grid = torch.zeros((self.num_bins, self.height, self.width))
        
        # Normalize voxel grid
        if self.normalize:
            voxel_grid = normalize_voxel_grid(voxel_grid)
        
        # Convert frame to tensor
        if frame is not None:
            frame = torch.from_numpy(frame).float()
            if frame.dim() == 2:
                frame = frame.unsqueeze(0)  # Add channel dimension
            frame = frame / 255.0  # Normalize to [0, 1]
        else:
            frame = torch.zeros((1, self.height, self.width))
        
        # Apply voxel augmentations
        if self.augment:
            voxel_grid, frame = self.voxel_augmentor(voxel_grid, frame)
        
        return {
            'voxel_grid': voxel_grid,
            'frame': frame,
            'name': sample_info['name'],
            'num_events': len(events) if events is not None else 0
        }
    
    def _load_events(self, sample_info: Dict) -> Optional[np.ndarray]:
        """Load events for a sample"""
        if 'event_file' in sample_info:
            # Load from single event file
            with h5py.File(sample_info['event_file'], 'r') as f:
                events = np.column_stack([
                    f['events']['x'][:],
                    f['events']['y'][:],
                    f['events']['t'][:],
                    f['events']['p'][:]
                ])
            return events
        
        elif 'events_file' in sample_info:
            # Load events within time window
            with h5py.File(sample_info['events_file'], 'r') as f:
                times = f['events']['t'][:]
                mask = (times >= sample_info['start_time']) & \
                       (times <= sample_info['end_time'])
                
                if np.any(mask):
                    events = np.column_stack([
                        f['events']['x'][mask],
                        f['events']['y'][mask],
                        f['events']['t'][mask],
                        f['events']['p'][mask]
                    ])
                    return events
        
        return None
    
    def _load_frame(self, sample_info: Dict) -> Optional[np.ndarray]:
        """Load ground truth frame"""
        if 'frame_file' in sample_info:
            import cv2
            frame = cv2.imread(sample_info['frame_file'], cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                # Resize if needed
                if frame.shape != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
            return frame
        
        return None


class EventSequenceDataset(EventDataset):
    """Dataset for loading sequences of events and frames"""
    
    def __init__(self,
                 root_dir: str,
                 sequence_length: int = 10,
                 overlap: int = 5,
                 **kwargs):
        """
        Initialize sequence dataset
        
        Args:
            root_dir: Root directory
            sequence_length: Number of frames per sequence
            overlap: Overlap between sequences
            **kwargs: Other arguments for EventDataset
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        super().__init__(root_dir, **kwargs)
        
        # Create sequences from samples
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[List[Dict]]:
        """Create overlapping sequences from samples"""
        sequences = []
        
        # Group samples by sequence/video
        grouped_samples = {}
        for sample in self.samples:
            key = sample.get('sequence', sample['name'].split('_')[0])
            if key not in grouped_samples:
                grouped_samples[key] = []
            grouped_samples[key].append(sample)
        
        # Create sequences with overlap
        for key, samples in grouped_samples.items():
            samples = sorted(samples, key=lambda x: x.get('frame_idx', 0))
            
            stride = self.sequence_length - self.overlap
            for i in range(0, len(samples) - self.sequence_length + 1, stride):
                sequences.append(samples[i:i + self.sequence_length])
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence of samples"""
        sequence = self.sequences[idx]
        
        voxel_grids = []
        frames = []
        
        for sample_info in sequence:
            # Load single sample
            sample_data = super().__getitem__(self.samples.index(sample_info))
            voxel_grids.append(sample_data['voxel_grid'])
            frames.append(sample_data['frame'])
        
        return {
            'voxel_grids': torch.stack(voxel_grids),  # [T, C, H, W]
            'frames': torch.stack(frames),  # [T, 1, H, W]
            'sequence_name': f"{sequence[0]['name']}_seq{idx}",
            'sequence_length': len(sequence)
        }


def create_dataloader(dataset_config: dict, 
                     split: str,
                     batch_size: int,
                     num_workers: int = 4,
                     shuffle: Optional[bool] = None) -> torch.utils.data.DataLoader:
    """
    Create dataloader for event dataset
    
    Args:
        dataset_config: Dataset configuration dict
        split: Data split ('train', 'val', 'test')
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (default: True for train)
        
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Create dataset
    dataset_class = EventSequenceDataset if dataset_config.get('use_sequences', False) else EventDataset
    dataset = dataset_class(
        root_dir=dataset_config['root_dir'],
        split=split,
        num_bins=dataset_config.get('num_bins', 5),
        height=dataset_config.get('height', 256),
        width=dataset_config.get('width', 256),
        augment=dataset_config.get('augment', True),
        normalize=dataset_config.get('normalize', True),
        dataset_type=dataset_config.get('dataset_type', 'generic')
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader

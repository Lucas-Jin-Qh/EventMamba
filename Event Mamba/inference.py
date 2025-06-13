#!/usr/bin/env python3
"""
EventMamba Inference Script
Performs inference on event streams for video reconstruction
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path
import h5py
import time

from models.eventmamba import EventMamba
from datasets.voxel_grid import event_to_voxel
from utils.visualization import Visualizer


class EventStreamProcessor:
    """Process event streams for inference"""
    
    def __init__(self, num_bins, height, width):
        self.num_bins = num_bins
        self.height = height
        self.width = width
    
    def process_events(self, events, time_window=None):
        """
        Process raw events into voxel grid
        
        Args:
            events: Array of events with shape (N, 4) [x, y, t, p]
            time_window: Optional time window in seconds
        
        Returns:
            voxel_grid: Tensor of shape (1, num_bins, H, W)
        """
        if len(events) == 0:
            return torch.zeros(1, self.num_bins, self.height, self.width)
        
        # Convert to voxel grid
        voxel_grid = event_to_voxel(
            events, 
            self.num_bins, 
            self.height, 
            self.width
        )
        
        # Add batch dimension
        voxel_grid = torch.from_numpy(voxel_grid).float().unsqueeze(0)
        
        return voxel_grid


class Inference:
    def __init__(self, checkpoint_path, config_path=None, device=None):
        """Initialize inference engine"""
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.checkpoint['config']
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self._setup_model()
        
        # Initialize event processor
        self.event_processor = EventStreamProcessor(
            num_bins=self.config['model']['num_bins'],
            height=self.config.get('inference', {}).get('height', 480),
            width=self.config.get('inference', {}).get('width', 640)
        )
        
        # Visualizer
        self.visualizer = Visualizer()
    
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
    
    def infer_single(self, voxel_grid, monte_carlo=True, K=8):
        """
        Perform inference on a single voxel grid
        
        Args:
            voxel_grid: Tensor of shape (1, num_bins, H, W)
            monte_carlo: Whether to use Monte Carlo inference
            K: Number of Monte Carlo samples
        
        Returns:
            pred_frame: Reconstructed frame tensor
        """
        voxel_grid = voxel_grid.to(self.device)
        
        with torch.no_grad():
            if monte_carlo:
                # Monte Carlo inference with random window offsets
                predictions = []
                for _ in range(K):
                    pred = self.model(voxel_grid)
                    predictions.append(pred)
                pred_frame = torch.stack(predictions).mean(dim=0)
            else:
                # Single forward pass
                pred_frame = self.model(voxel_grid)
        
        return pred_frame
    
    def process_event_file(self, event_file, output_path, **kwargs):
        """
        Process an event file and save reconstructed video
        
        Args:
            event_file: Path to event file (h5, npy, or txt format)
            output_path: Path to save output video
            **kwargs: Additional arguments for processing
        """
        print(f"Processing event file: {event_file}")
        
        # Load events
        events = self._load_events(event_file)
        print(f"Loaded {len(events)} events")
        
        # Process options
        time_window = kwargs.get('time_window', 0.05)  # 50ms default
        overlap = kwargs.get('overlap', 0.5)  # 50% overlap
        monte_carlo = kwargs.get('monte_carlo', True)
        K = kwargs.get('K', 8)
        fps = kwargs.get('fps', 30)
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = None
        
        # Process events in sliding windows
        start_time = events[0, 2]
        end_time = events[-1, 2]
        current_time = start_time
        
        frames = []
        timestamps = []
        
        progress_bar = tqdm(total=int((end_time - start_time) / (time_window * (1 - overlap))))
        
        while current_time < end_time - time_window:
            # Extract events in current window
            window_end = current_time + time_window
            mask = (events[:, 2] >= current_time) & (events[:, 2] < window_end)
            window_events = events[mask]
            
            if len(window_events) > 0:
                # Convert to voxel grid
                voxel_grid = self.event_processor.process_events(window_events)
                
                # Perform inference
                pred_frame = self.infer_single(voxel_grid, monte_carlo, K)
                
                # Convert to image
                frame_np = self._tensor_to_image(pred_frame[0, 0])
                
                # Initialize video writer with frame size
                if out_video is None:
                    h, w = frame_np.shape[:2]
                    out_video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                
                # Write frame
                out_video.write(frame_np)
                frames.append(frame_np)
                timestamps.append(current_time)
            
            # Move to next window
            current_time += time_window * (1 - overlap)
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Release video writer
        if out_video is not None:
            out_video.release()
        
        print(f"Saved reconstructed video to: {output_path}")
        print(f"Total frames: {len(frames)}")
        
        # Save additional outputs if requested
        if kwargs.get('save_frames', False):
            self._save_frames(frames, timestamps, output_dir)
        
        return frames, timestamps
    
    def process_event_stream(self, event_generator, output_path, **kwargs):
        """
        Process a live event stream
        
        Args:
            event_generator: Generator yielding event batches
            output_path: Path to save output video
            **kwargs: Additional arguments for processing
        """
        print("Processing live event stream...")
        
        # Process options
        buffer_size = kwargs.get('buffer_size', 10000)
        time_window = kwargs.get('time_window', 0.05)
        monte_carlo = kwargs.get('monte_carlo', False)  # Faster for real-time
        display = kwargs.get('display', True)
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = kwargs.get('fps', 30)
        out_video = None
        
        # Event buffer
        event_buffer = []
        last_process_time = None
        frame_count = 0
        
        try:
            for events in event_generator:
                # Add to buffer
                event_buffer.extend(events)
                
                # Keep buffer size manageable
                if len(event_buffer) > buffer_size * 2:
                    event_buffer = event_buffer[-buffer_size:]
                
                # Process when enough time has passed
                if len(event_buffer) > 0:
                    current_time = event_buffer[-1][2]
                    
                    if last_process_time is None:
                        last_process_time = current_time
                    
                    if current_time - last_process_time >= time_window:
                        # Convert buffer to array
                        events_array = np.array(event_buffer)
                        
                        # Extract recent events
                        mask = events_array[:, 2] >= (current_time - time_window)
                        window_events = events_array[mask]
                        
                        if len(window_events) > 0:
                            # Process events
                            voxel_grid = self.event_processor.process_events(window_events)
                            
                            # Inference
                            start_inference = time.time()
                            pred_frame = self.infer_single(voxel_grid, monte_carlo, K=1)
                            inference_time = time.time() - start_inference
                            
                            # Convert to image
                            frame_np = self._tensor_to_image(pred_frame[0, 0])
                            
                            # Initialize video writer
                            if out_video is None:
                                h, w = frame_np.shape[:2]
                                out_video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                            
                            # Write frame
                            out_video.write(frame_np)
                            frame_count += 1
                            
                            # Display if requested
                            if display:
                                # Add info overlay
                                info_frame = frame_np.copy()
                                cv2.putText(info_frame, f"Frame: {frame_count}", (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(info_frame, f"Inference: {inference_time*1000:.1f}ms", 
                                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(info_frame, f"Events: {len(window_events)}", (10, 90),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                
                                cv2.imshow("EventMamba Reconstruction", info_frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            
                            last_process_time = current_time
                
        except KeyboardInterrupt:
            print("\nStream processing interrupted by user")
        finally:
            # Cleanup
            if out_video is not None:
                out_video.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"Processed {frame_count} frames")
            print(f"Saved video to: {output_path}")
    
    def batch_inference(self, input_dir, output_dir, **kwargs):
        """
        Process multiple event files in batch
        
        Args:
            input_dir: Directory containing event files
            output_dir: Directory to save output videos
            **kwargs: Additional arguments for processing
        """
        # Find all event files
        input_path = Path(input_dir)
        event_files = list(input_path.glob('*.h5')) + \
                     list(input_path.glob('*.npy')) + \
                     list(input_path.glob('*.txt'))
        
        print(f"Found {len(event_files)} event files")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each file
        for event_file in tqdm(event_files, desc="Processing files"):
            # Generate output path
            output_name = event_file.stem + '_reconstructed.mp4'
            output_path = os.path.join(output_dir, output_name)
            
            try:
                self.process_event_file(str(event_file), output_path, **kwargs)
            except Exception as e:
                print(f"Error processing {event_file}: {e}")
                continue
        
        print(f"Batch processing completed. Results saved to: {output_dir}")
    
    def _load_events(self, event_file):
        """Load events from various file formats"""
        ext = os.path.splitext(event_file)[1].lower()
        
        if ext == '.h5' or ext == '.hdf5':
            # HDF5 format
            with h5py.File(event_file, 'r') as f:
                events = np.column_stack([
                    f['events']['x'][:],
                    f['events']['y'][:],
                    f['events']['t'][:],
                    f['events']['p'][:]
                ])
        elif ext == '.npy':
            # NumPy format
            events = np.load(event_file)
        elif ext == '.txt' or ext == '.csv':
            # Text format
            events = np.loadtxt(event_file, delimiter=',')
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return events
    
    def _tensor_to_image(self, tensor):
        """Convert tensor to OpenCV image"""
        # Ensure tensor is on CPU and convert to numpy
        image = tensor.cpu().numpy()
        
        # Normalize to [0, 255]
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        # Convert grayscale to BGR for OpenCV
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image
    
    def _save_frames(self, frames, timestamps, output_dir):
        """Save individual frames and timestamps"""
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save frames
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            frame_path = os.path.join(frames_dir, f'frame_{i:06d}.png')
            cv2.imwrite(frame_path, frame)
        
        # Save timestamps
        timestamps_path = os.path.join(output_dir, 'timestamps.npy')
        np.save(timestamps_path, np.array(timestamps))
        
        print(f"Saved {len(frames)} frames to: {frames_dir}")


def main():
    parser = argparse.ArgumentParser(description='EventMamba inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True,
                        help='Input event file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output video file or directory')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'batch', 'stream'],
                        help='Inference mode')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    # Processing options
    parser.add_argument('--time-window', type=float, default=0.05,
                        help='Time window for event aggregation (seconds)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap ratio for sliding window')
    parser.add_argument('--monte-carlo', action='store_true',
                        help='Use Monte Carlo inference')
    parser.add_argument('--K', type=int, default=8,
                        help='Number of Monte Carlo samples')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS')
    parser.add_argument('--save-frames', action='store_true',
                        help='Save individual frames')
    parser.add_argument('--display', action='store_true',
                        help='Display live reconstruction (stream mode)')
    
    args = parser.parse_args()
    
    # Create inference engine
    inference = Inference(args.checkpoint, args.config, args.device)
    
    # Prepare kwargs
    kwargs = {
        'time_window': args.time_window,
        'overlap': args.overlap,
        'monte_carlo': args.monte_carlo,
        'K': args.K,
        'fps': args.fps,
        'save_frames': args.save_frames,
        'display': args.display
    }
    
    # Run inference based on mode
    if args.mode == 'single':
        inference.process_event_file(args.input, args.output, **kwargs)
    elif args.mode == 'batch':
        inference.batch_inference(args.input, args.output, **kwargs)
    elif args.mode == 'stream':
        # For stream mode, you would need to implement an event generator
        # based on your specific hardware/data source
        raise NotImplementedError("Stream mode requires custom event generator implementation")


if __name__ == '__main__':
    main()

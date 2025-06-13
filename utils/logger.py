"""
日志记录工具
支持控制台、文件、TensorBoard和WandB等多种日志后端
"""

import logging
import json
import csv
import time
from pathlib import Path
from typing import Dict, Optional, List, Union, Any
from datetime import datetime
import numpy as np
import torch

# 尝试导入可选的依赖
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def setup_logger(
    name: str,
    log_dir: str,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        level: 日志级别
        format_str: 日志格式字符串
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.hasHandlers():
        return logger
    
    # 默认格式
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_str)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_dir / f'{name}_{timestamp}.log'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取已存在的logger"""
    return logging.getLogger(name)


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, float],
    step: int,
    prefix: str = 'train'
) -> None:
    """
    记录指标到logger
    
    Args:
        logger: 日志记录器
        metrics: 指标字典
        step: 当前步数
        prefix: 指标前缀
    """
    # 构建日志消息
    msg_parts = [f"Step {step}"]
    for key, value in metrics.items():
        if isinstance(value, float):
            msg_parts.append(f"{prefix}/{key}: {value:.4f}")
        else:
            msg_parts.append(f"{prefix}/{key}: {value}")
    
    logger.info(" | ".join(msg_parts))


class BaseLogger:
    """日志记录器基类"""
    
    def __init__(self, log_dir: str, exp_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name = exp_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        raise NotImplementedError
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """记录多个标量值"""
        for key, value in values.items():
            self.log_scalar(f"{tag}/{key}", value, step)
    
    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int):
        """记录图像"""
        raise NotImplementedError
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """记录直方图"""
        raise NotImplementedError
    
    def log_text(self, tag: str, text: str, step: int):
        """记录文本"""
        raise NotImplementedError
    
    def close(self):
        """关闭日志记录器"""
        pass


class TensorBoardLogger(BaseLogger):
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir: str, exp_name: Optional[str] = None):
        super().__init__(log_dir, exp_name)
        
        if not HAS_TENSORBOARD:
            raise ImportError("TensorBoard not installed. Run: pip install tensorboard")
        
        self.writer = SummaryWriter(self.log_dir / self.exp_name)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        self.writer.add_scalar(tag, value, step)
    
    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int):
        """记录图像"""
        if isinstance(image, np.ndarray):
            # 转换为CHW格式
            if image.ndim == 2:
                image = np.expand_dims(image, 0)
            elif image.ndim == 3 and image.shape[2] in [1, 3, 4]:
                image = np.transpose(image, (2, 0, 1))
        
        self.writer.add_image(tag, image, step)
    
    def log_images(self, tag: str, images: Union[torch.Tensor, np.ndarray], step: int):
        """记录多个图像"""
        self.writer.add_images(tag, images, step)
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """记录直方图"""
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """记录文本"""
        self.writer.add_text(tag, text, step)
    
    def log_graph(self, model: torch.nn.Module, input_shape: Tuple[int, ...]):
        """记录模型结构"""
        dummy_input = torch.zeros(input_shape)
        self.writer.add_graph(model, dummy_input)
    
    def close(self):
        """关闭写入器"""
        self.writer.close()


class WandBLogger(BaseLogger):
    """Weights & Biases日志记录器"""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ):
        if not HAS_WANDB:
            raise ImportError("WandB not installed. Run: pip install wandb")
        
        # 初始化WandB
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags
        )
        
        super().__init__(wandb.run.dir, name)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        wandb.log({tag: value}, step=step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """记录多个标量值"""
        log_dict = {f"{tag}/{key}": value for key, value in values.items()}
        wandb.log(log_dict, step=step)
    
    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int):
        """记录图像"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # 转换为HWC格式
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        elif image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            image = np.transpose(image, (1, 2, 0))
        
        wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """记录直方图"""
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        
        wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    def log_text(self, tag: str, text: str, step: int):
        """记录文本"""
        wandb.log({tag: text}, step=step)
    
    def log_table(self, tag: str, data: List[List[Any]], columns: List[str]):
        """记录表格数据"""
        table = wandb.Table(data=data, columns=columns)
        wandb.log({tag: table})
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """监控模型参数和梯度"""
        wandb.watch(model, log_freq=log_freq)
    
    def close(self):
        """结束运行"""
        wandb.finish()


class CSVLogger(BaseLogger):
    """CSV日志记录器"""
    
    def __init__(self, log_dir: str, filename: str = 'metrics.csv'):
        super().__init__(log_dir)
        
        self.csv_path = self.log_dir / filename
        self.metrics_buffer = []
        self.headers = ['step']
        self.header_written = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        self.metrics_buffer.append({
            'step': step,
            tag: value
        })
        
        # 更新headers
        if tag not in self.headers:
            self.headers.append(tag)
        
        # 定期写入
        if len(self.metrics_buffer) >= 10:
            self.flush()
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """记录多个标量值"""
        row = {'step': step}
        for key, value in values.items():
            full_tag = f"{tag}/{key}"
            row[full_tag] = value
            if full_tag not in self.headers:
                self.headers.append(full_tag)
        
        self.metrics_buffer.append(row)
        
        if len(self.metrics_buffer) >= 10:
            self.flush()
    
    def flush(self):
        """将缓冲区数据写入CSV文件"""
        if not self.metrics_buffer:
            return
        
        # 写入CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            
            # 写入header（仅第一次）
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            
            # 写入数据
            for row in self.metrics_buffer:
                # 填充缺失的字段
                for header in self.headers:
                    if header not in row:
                        row[header] = ''
                writer.writerow(row)
        
        # 清空缓冲区
        self.metrics_buffer = []
    
    def close(self):
        """关闭并写入剩余数据"""
        self.flush()
    
    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int):
        """CSV不支持图像记录"""
        pass
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """记录直方图统计信息"""
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        
        # 计算统计信息
        stats = {
            f"{tag}/mean": np.mean(values),
            f"{tag}/std": np.std(values),
            f"{tag}/min": np.min(values),
            f"{tag}/max": np.max(values)
        }
        
        self.log_scalars(tag, stats, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """CSV不支持文本记录"""
        pass


class MultiLogger:
    """多后端日志记录器"""
    
    def __init__(self, loggers: List[BaseLogger]):
        self.loggers = loggers
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值到所有后端"""
        for logger in self.loggers:
            logger.log_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """记录多个标量值"""
        for logger in self.loggers:
            logger.log_scalars(tag, values, step)
    
    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int):
        """记录图像"""
        for logger in self.loggers:
            logger.log_image(tag, image, step)
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """记录直方图"""
        for logger in self.loggers:
            logger.log_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """记录文本"""
        for logger in self.loggers:
            logger.log_text(tag, text, step)
    
    def close(self):
        """关闭所有日志记录器"""
        for logger in self.loggers:
            logger.close()


class TrainingLogger:
    """训练专用日志记录器"""
    
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_csv: bool = True,
        use_wandb: bool = False,
        wandb_config: Optional[Dict] = None
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志记录器列表
        loggers = []
        
        if use_tensorboard and HAS_TENSORBOARD:
            loggers.append(TensorBoardLogger(log_dir))
        
        if use_csv:
            loggers.append(CSVLogger(log_dir))
        
        if use_wandb and HAS_WANDB:
            wandb_config = wandb_config or {}
            loggers.append(WandBLogger(**wandb_config))
        
        self.logger = MultiLogger(loggers) if loggers else None
        
        # 训练统计
        self.epoch_start_time = None
        self.training_start_time = time.time()
    
    def log_epoch_start(self, epoch: int):
        """记录epoch开始"""
        self.epoch_start_time = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {epoch} started")
        print(f"{'='*50}")
    
    def log_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """记录epoch结束"""
        epoch_time = time.time() - self.epoch_start_time
        
        # 打印到控制台
        print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Train metrics: {train_metrics}")
        if val_metrics:
            print(f"Val metrics: {val_metrics}")
        
        # 记录到日志
        if self.logger:
            self.logger.log_scalars('train', train_metrics, epoch)
            if val_metrics:
                self.logger.log_scalars('val', val_metrics, epoch)
            self.logger.log_scalar('time/epoch', epoch_time, epoch)
    
    def log_training_end(self):
        """记录训练结束"""
        total_time = time.time() - self.training_start_time
        print(f"\n{'='*50}")
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"{'='*50}")
        
        if self.logger:
            self.logger.close()

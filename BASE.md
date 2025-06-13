# EventMamba 代码框架设计

## 1. 项目结构
```
EventMamba/
├── configs/
│   ├── default_config.yaml
│   ├── dataset_config.yaml
│   └── model_config.yaml
├── datasets/
│   ├── __init__.py
│   ├── event_dataset.py
│   ├── voxel_grid.py
│   └── data_augmentation.py
├── models/
│   ├── __init__.py
│   ├── eventmamba.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── rwo_mamba.py
│   │   ├── hsfc_mamba.py
│   │   ├── mamba_block.py
│   │   └── ssm_layer.py
│   ├── encoder.py
│   ├── decoder.py
│   └── utils.py
├── losses/
│   ├── __init__.py
│   ├── lpips_loss.py
│   ├── temporal_consistency.py
│   └── combined_loss.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── visualization.py
│   └── logger.py
├── train.py
├── test.py
└── inference.py
```

## 2. 核心模块架构图

### 2.1 整体架构流程
```
事件流输入 (N events)
    ↓
[事件表示模块]
    ├── 事件分组 (根据时间戳)
    ├── 体素网格生成 (H×W×B)
    └── 时间归一化
    ↓
[U-Net 主体架构]
    ├── 编码器路径
    │   ├── Conv3D (初始特征提取)
    │   ├── RWOMamba Block 1 (下采样)
    │   ├── HSFCMamba Block 1
    │   ├── RWOMamba Block 2 (下采样)
    │   ├── HSFCMamba Block 2
    │   └── ... (继续下采样)
    │
    ├── 瓶颈层
    │   ├── RWOMamba Block
    │   └── HSFCMamba Block
    │
    └── 解码器路径
        ├── RWOMamba Block 1 (上采样)
        ├── HSFCMamba Block 1
        ├── Skip Connection 融合
        ├── RWOMamba Block 2 (上采样)
        ├── HSFCMamba Block 2
        └── ... (继续上采样)
    ↓
[输出层]
    └── Conv2D → 重建的强度图像
```

### 2.2 RWOMamba 模块详细设计
```python
class RWOMamba:
    """随机窗口偏移Mamba模块"""
    
    构造函数参数:
    - embed_dim: 嵌入维度
    - window_size: 窗口大小
    - num_heads: 注意力头数
    - ssm_ratio: SSM扩展比率
    
    主要组件:
    1. 随机窗口偏移生成器
       - 训练时: 随机采样偏移量 (δx, δy) ~ U(0, window_size)
       - 测试时: 蒙特卡洛采样 (K=8次)
    
    2. Vision Mamba Block (VMB)
       - LayerNorm
       - Linear投影
       - DWConv (深度可分离卷积)
       - SSM层 (选择性状态空间模型)
       - 输出投影
    
    3. 窗口分区与还原
       - window_partition()
       - window_reverse()
```

### 2.3 HSFCMamba 模块详细设计
```python
class HSFCMamba:
    """希尔伯特空间填充曲线Mamba模块"""
    
    主要组件:
    1. Hilbert曲线扫描器
       - hilbert_scan(): 3D→1D序列化
       - trans_hilbert_scan(): 反向扫描
    
    2. 双向处理
       - Forward路径: Hilbert扫描
       - Backward路径: Trans-Hilbert扫描
       - 特征融合: concat + projection
    
    3. VMB处理
       - 序列化特征 → VMB → 反序列化
```

### 2.4 State Space Model (SSM) 核心
```python
class SelectiveSSM:
    """选择性状态空间模型"""
    
    核心方程:
    - 离散化: Δ, A_bar, B_bar = discretize(A, B, C, Δ)
    - 状态更新: h[k] = A_bar * h[k-1] + B_bar * x[k]
    - 输出: y[k] = C * h[k]
    
    关键特性:
    - 输入依赖的参数 (Δ, B, C)
    - 硬件感知的并行扫描
    - 线性时间复杂度 O(L)
```

## 3. 数据处理流程

### 3.1 事件表示转换
```python
def event_to_voxel(events, num_bins, height, width):
    """
    将事件流转换为体素网格
    
    输入:
    - events: [(x, y, t, p), ...] 事件列表
    - num_bins: 时间维度的bin数量
    - height, width: 空间维度
    
    输出:
    - voxel_grid: [B, H, W] 体素网格
    """
    # 1. 时间归一化到[0, num_bins-1]
    # 2. 双线性插值分配极性
    # 3. 累积到体素网格
```

### 3.2 数据增强策略
```python
class EventAugmentation:
    - 随机时间翻转
    - 随机空间翻转
    - 随机噪声注入
    - 随机事件丢弃
    - 对比度阈值扰动
```

## 4. 损失函数设计

### 4.1 组合损失
```python
class CombinedLoss:
    def __init__(self, lambda_l1=20, lambda_lpips=2, lambda_tc=0.5):
        self.l1_loss = nn.L1Loss()
        self.lpips_loss = LPIPS(net='vgg')
        self.tc_loss = TemporalConsistencyLoss()
    
    def forward(self, pred, gt, flow=None):
        loss_l1 = self.lambda_l1 * self.l1_loss(pred, gt)
        loss_lpips = self.lambda_lpips * self.lpips_loss(pred, gt)
        loss_tc = self.lambda_tc * self.tc_loss(pred, flow) if flow else 0
        return loss_l1 + loss_lpips + loss_tc
```

## 5. 训练策略

### 5.1 训练配置
```yaml
training:
  batch_size: 4
  patch_size: 256
  num_epochs: 400
  optimizer:
    type: AdamW
    lr: 1e-4
    weight_decay: 0.01
  scheduler:
    type: ExponentialLR
    gamma: 0.99
  
model:
  base_channel: 32
  num_stages: 4
  window_size: 8
  ssm_ratio: 2
  num_bins: 5
```

### 5.2 训练循环伪代码
```python
def train_epoch(model, dataloader, optimizer, criterion):
    for batch in dataloader:
        # 1. 事件转体素
        voxel_grid = event_to_voxel(batch['events'])
        
        # 2. 前向传播
        pred_frames = model(voxel_grid)
        
        # 3. 计算损失
        loss = criterion(pred_frames, batch['gt_frames'])
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 6. 推理优化

### 6.1 蒙特卡洛推理
```python
def monte_carlo_inference(model, voxel_grid, K=8):
    """
    使用K次随机窗口偏移进行推理
    """
    predictions = []
    for _ in range(K):
        pred = model(voxel_grid)
        predictions.append(pred)
    return torch.mean(torch.stack(predictions), dim=0)
```

### 6.2 批处理加速
```python
def batched_inference(model, voxel_grid, K=8):
    """
    批处理蒙特卡洛推理以加速
    """
    # 复制输入K次
    batch_input = voxel_grid.repeat(K, 1, 1, 1)
    # 单次前向传播
    batch_output = model(batch_input)
    # 平均K个结果
    return batch_output.view(K, -1, *batch_output.shape[1:]).mean(0)
```

## 7. 关键实现细节

### 7.1 Hilbert曲线生成
```python
def generate_hilbert_curve(order):
    """生成指定阶数的Hilbert曲线坐标"""
    # 递归生成Hilbert曲线
    # 返回坐标序列
```

### 7.2 窗口分区与偏移
```python
def window_partition_with_offset(x, window_size, offset):
    """带偏移的窗口分区"""
    # 1. 应用偏移
    # 2. 分区
    # 3. 返回窗口张量
```

### 7.3 高效SSM实现
```python
def selective_scan(x, delta, A, B, C):
    """选择性扫描算法"""
    # 使用CUDA优化的并行扫描
    # 支持变长序列
```

## 8. 评估指标
```python
class Metrics:
    - MSE (均方误差)
    - SSIM (结构相似性)
    - LPIPS (感知相似性)
    - 时间一致性误差
```

## 9. 可视化工具
```python
class Visualizer:
    - 事件流可视化
    - 体素网格可视化
    - 重建结果对比
    - 训练曲线绘制
```

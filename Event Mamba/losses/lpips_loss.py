"""
LPIPS (Learned Perceptual Image Patch Similarity) 损失实现
基于预训练的VGG或AlexNet网络提取感知特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from collections import OrderedDict


class VGGFeatureExtractor(nn.Module):
    """
    VGG特征提取器，用于LPIPS损失计算
    """
    
    def __init__(self, requires_grad: bool = False):
        super().__init__()
        
        # 使用torchvision的预训练VGG16
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True)
        except:
            # 如果无法下载预训练权重，使用随机初始化
            import torchvision.models as models
            vgg = models.vgg16(pretrained=False)
            print("Warning: Using randomly initialized VGG16 for LPIPS")
        
        # 选择特定层作为特征提取
        self.layers = {
            '3': 'relu1_2',   # 64 channels
            '8': 'relu2_2',   # 128 channels
            '15': 'relu3_3',  # 256 channels
            '22': 'relu4_3',  # 512 channels
            '29': 'relu5_3'   # 512 channels
        }
        
        # 构建特征提取网络
        features = OrderedDict()
        for name, module in vgg.features._modules.items():
            features[name] = module
            if name in self.layers:
                # 在指定层后截断
                break
        
        self.features = nn.Sequential(features)
        
        # 冻结参数
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # 注册中间特征的hook
        self.feature_maps = {}
        for name, layer in self.features._modules.items():
            if name in self.layers:
                layer.register_forward_hook(self._get_features(name))
    
    def _get_features(self, name: str):
        def hook(module, input, output):
            self.feature_maps[self.layers[name]] = output
        return hook
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取多尺度特征
        Args:
            x: (B, C, H, W) 输入图像
        Returns:
            features: 各层特征列表
        """
        # 清空特征字典
        self.feature_maps = {}
        
        # 如果输入是灰度图，复制到3通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # 前向传播
        _ = self.features(x)
        
        # 返回有序的特征列表
        features = []
        for layer_name in ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']:
            if layer_name in self.feature_maps:
                features.append(self.feature_maps[layer_name])
        
        return features


class LPIPS(nn.Module):
    """
    LPIPS损失模块
    """
    
    def __init__(
        self,
        net: str = 'vgg',
        use_dropout: bool = False,
        eval_mode: bool = True,
        normalize: bool = True,
        reduction: str = 'mean'
    ):
        """
        Args:
            net: 特征提取网络类型 ('vgg', 'alex')
            use_dropout: 是否使用dropout
            eval_mode: 是否设置为评估模式
            normalize: 是否归一化输入
            reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.net_type = net
        self.normalize = normalize
        self.reduction = reduction
        
        # 创建特征提取器
        if net == 'vgg':
            self.feature_extractor = VGGFeatureExtractor(requires_grad=False)
            # VGG各层的通道数
            self.channels = [64, 128, 256, 512, 512]
        else:
            raise NotImplementedError(f"Network {net} not implemented")
        
        # 为每层创建1x1卷积作为线性层
        self.lins = nn.ModuleList()
        for ch in self.channels:
            lin = nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.lins.append(lin)
        
        # 初始化权重
        self._initialize_weights()
        
        # 设置评估模式
        if eval_mode:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        
        # 图像归一化参数（ImageNet统计）
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def _initialize_weights(self):
        """初始化1x1卷积权重"""
        # 使用经验值初始化（这些值通常从原始LPIPS论文中获得）
        default_weights = {
            'vgg': [1.0, 1.0, 1.0, 1.0, 1.0],  # 简化的权重
        }
        
        weights = default_weights.get(self.net_type, [1.0] * len(self.lins))
        
        for lin, w in zip(self.lins, weights):
            lin.weight.data.fill_(w / lin.weight.shape[1])  # 除以通道数进行归一化
    
    def normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        归一化输入张量
        """
        if x.shape[1] == 1:
            # 灰度图转RGB
            x = x.repeat(1, 3, 1, 1)
        
        if self.normalize:
            x = (x - self.mean) / self.std
        
        return x
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算LPIPS损失
        Args:
            pred: (B, C, H, W) 预测图像
            target: (B, C, H, W) 目标图像
        Returns:
            loss: LPIPS损失值
        """
        # 确保输入在[0, 1]范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # 归一化输入
        pred_norm = self.normalize_tensor(pred)
        target_norm = self.normalize_tensor(target)
        
        # 提取特征
        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        
        # 计算每层的感知距离
        diffs = []
        for pred_feat, target_feat, lin in zip(pred_features, target_features, self.lins):
            # 特征归一化
            pred_feat_norm = pred_feat / (pred_feat.norm(dim=1, keepdim=True) + 1e-10)
            target_feat_norm = target_feat / (target_feat.norm(dim=1, keepdim=True) + 1e-10)
            
            # 计算差异
            diff = (pred_feat_norm - target_feat_norm) ** 2
            
            # 应用1x1卷积（学习的权重）
            weighted_diff = lin(diff)
            
            # 空间平均
            diff_spatial_mean = weighted_diff.mean(dim=(2, 3))
            diffs.append(diff_spatial_mean)
        
        # 合并所有层的损失
        loss = torch.cat(diffs, dim=1).sum(dim=1)
        
        # 应用reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def lpips_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = 'vgg',
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    便捷函数：计算LPIPS损失
    
    Args:
        pred: 预测图像
        target: 目标图像
        net: 网络类型
        reduction: 损失聚合方式
    
    Returns:
        loss: LPIPS损失值
    """
    # 创建LPIPS模块（通常应该在外部创建并重用）
    lpips_module = LPIPS(net=net, reduction=reduction)
    lpips_module = lpips_module.to(pred.device)
    
    with torch.no_grad():
        loss = lpips_module(pred, target)
    
    return loss


class MultiScaleLPIPS(nn.Module):
    """
    多尺度LPIPS损失
    在多个分辨率上计算LPIPS损失
    """
    
    def __init__(
        self,
        scales: List[float] = [1.0, 0.5, 0.25],
        weights: Optional[List[float]] = None,
        net: str = 'vgg'
    ):
        super().__init__()
        
        self.scales = scales
        self.weights = weights if weights is not None else [1.0] * len(scales)
        
        # 为每个尺度创建LPIPS模块
        self.lpips_modules = nn.ModuleList([
            LPIPS(net=net) for _ in scales
        ])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算多尺度LPIPS损失
        """
        total_loss = 0
        
        for scale, weight, lpips_module in zip(self.scales, self.weights, self.lpips_modules):
            if scale != 1.0:
                # 下采样到指定尺度
                pred_scaled = F.interpolate(
                    pred,
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False
                )
                target_scaled = F.interpolate(
                    target,
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                pred_scaled = pred
                target_scaled = target
            
            # 计算该尺度的LPIPS损失
            loss = lpips_module(pred_scaled, target_scaled)
            total_loss += weight * loss
        
        return total_loss

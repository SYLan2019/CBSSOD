import torch
import torch.nn.functional as F

def l2_loss(device, scale=1.0, x1 = None, x2 = None):
    """
    构造一个伪造的 L2 损失，用于替代真实损失（如 loss 为 0 时防止异常）

    Args:
        device (torch.device): 生成 loss 的设备
        size (int): 向量维度
        scale (float): 缩放系数，控制 loss 大小

    Returns:
        torch.Tensor: 伪造的 L2 损失标量
    """
    x = torch.rand(32, device=device)
    y = torch.rand(32, device=device).detach()
    return F.mse_loss(x, y) * scale
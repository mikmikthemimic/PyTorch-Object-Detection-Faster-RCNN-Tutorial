import torch
from torch import nn

class CoordinateAttention(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=32):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden_dim = max(8, in_dim // reduction)
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        b,c,h,w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose(-1, -2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(-1, -2)
        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)
        out = identity * a_h * a_w
        return out
    
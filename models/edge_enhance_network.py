import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings

warnings.filterwarnings("ignore")

# --- 辅助模块 ---

# patch merging 模块，用于降低特征图的空间分辨率，同时增加通道数
class patchmerging(nn.Module):
    """
    Patch Merging层。
    将一个[B, C, H, W]的张量转换为[B, C, H/2, W/2]的张量。
    """
    def __init__(self, ch_in):
        super(patchmerging, self).__init__()
        # 线性层，将4倍的输入通道数降维回原始通道数
        self.reduction = nn.Linear(ch_in * 4, ch_in)
        # 层归一化
        self.norm = nn.LayerNorm(ch_in * 4)

    def forward(self, x):
        """
        :param x: 输入张量，形状为 [B, C, H, W]。
        :return: 输出张量，形状为 [B, C, H//2, W//2]。
        """
        B, C, H, W = x.shape
        # 将输入张量在空间维度上分块
        x0 = x[:, :, 0::2, 0::2]  # 左上角
        x1 = x[:, :, 1::2, 0::2]  # 左下角
        x2 = x[:, :, 0::2, 1::2]  # 右上角
        x3 = x[:, :, 1::2, 1::2]  # 右下角

        # 沿着通道维度连接分块后的张量 [B, 4*C, H/2, W/2]
        out = torch.cat([x0, x1, x2, x3], dim=1)

        # 重塑张量以进行线性变换 [B, 4*C, (H/2)*(W/2)] -> [B, (H/2)*(W/2), 4*C]
        out = out.view(B, C * 4, -1).permute(0, 2, 1)

        # 应用层归一化和线性降维
        out = self.norm(out)
        out = self.reduction(out)  # [B, HW/4, C]

        # 恢复张量形状 [B, C, (H/2)*(W/2)] -> [B, C, H/2, W/2]
        out = out.permute(0, 2, 1).view(B, C, H // 2, W // 2)
        return out

# 通道注意力（Channel Attention）模块
class CA(nn.Module):
    def __init__(self, ch_in, ch_out=None, b=1, gama=2):
        super(CA, self).__init__()
        # 根据输入通道数动态计算一维卷积的核大小
        kernel_size = int(abs((math.log(ch_in, 2) + b) / gama))
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1

        padding = kernel_size // 2

        # 全局自适应平均池化层
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # 一维卷积层，Torch的Conv1d输入为 (B, C_in, L)，这里把Feature Channel当作L
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        # Sigmoid激活函数
        self.active = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # 全局平均池化
        y = self.pool(x)  # [B, C, 1, 1]

        # 重塑以适应一维卷积 [B, 1, C]
        y = y.view(b, 1, c)

        # 一维卷积
        y = self.conv(y)  # [B, 1, C]

        # 重塑回原始形状 [B, C, 1, 1]
        y = y.view(b, c, 1, 1)

        # 生成注意力权重并应用
        y = self.active(y)
        return x * y

# 边缘特征提取模块1 (Edge Feature Extraction 1)
class Conv1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv1, self).__init__()

        self.conv11 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )

        self.conv112 = nn.Sequential(
            nn.Conv2d(ch_in * 3 // 2, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.conv113 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.conv31 = nn.Sequential(
            nn.Conv2d(ch_in // 2, ch_in // 2, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(ch_in // 2, ch_in // 2, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )

        self.conv33 = nn.Sequential(
            nn.Conv2d(ch_in // 2, ch_in // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )
        self.CA = CA(ch_in // 2, ch_in // 2, b=1, gama=2)

    def forward(self, x):
        x1 = self.conv113(x)  # 残差连接
        x2 = self.conv11(x)
        y = self.conv31(x2)
        y = self.CA(y)
        z = self.conv13(x2)
        z = self.CA(z)
        w = self.conv33(x2)
        out = torch.cat([y, z, w], dim=1)
        out = self.conv112(out)
        out = out + x1  # 添加残差
        return out

# 边缘特征提取模块2 (Edge Feature Extraction 2)
class Conv2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv2, self).__init__()

        self.conv11 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )

        self.conv112 = nn.Sequential(
            nn.Conv2d(3 * ch_in // 2, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        # 注意：Conv2 中没有 conv113，x 直接作为残差，前提是通道数匹配
        # 如果通道数不匹配，通常需要处理，但根据代码上下文，这里主要用于 channel_in == channel_out

        self.conv31 = nn.Sequential(
            nn.Conv2d(ch_in // 2, ch_in // 2, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(ch_in // 2, ch_in // 2, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )

        self.conv33 = nn.Sequential(
            nn.Conv2d(ch_in // 2, ch_in // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )
        self.CA = CA(ch_in // 2, ch_in // 2, b=1, gama=2)

    def forward(self, x):
        x1 = x  # 残差连接
        x2 = self.conv11(x)
        y = self.conv31(x2)
        y = self.CA(y)
        z = self.conv13(x2)
        z = self.CA(z)
        w = self.conv33(x2)
        out = torch.cat([y, z, w], dim=1)
        out = self.conv112(out)
        out = out + x1  # 添加残差
        return out

# 编码器块
class encoder_block2(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(encoder_block2, self).__init__()
        self.block1 = Conv1(ch_in=channel_in, ch_out=channel_out)
        self.block2 = Conv2(ch_in=channel_out, ch_out=channel_out)
        self.block3 = Conv2(ch_in=channel_out, ch_out=channel_out)
        self.pool = patchmerging(channel_out)
    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.pool(y)
        return y

class encoder_block3(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(encoder_block3, self).__init__()
        self.block1 = Conv1(ch_in=channel_in, ch_out=channel_out)
        self.block2 = Conv2(ch_in=channel_out, ch_out=channel_out)
        self.block3 = Conv2(ch_in=channel_out, ch_out=channel_out)
        self.block4 = Conv2(ch_in=channel_out, ch_out=channel_out)
        self.pool = patchmerging(channel_out)
    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.pool(y)
        return y

class encoder_block4(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(encoder_block4, self).__init__()
        self.block1 = Conv1(ch_in=channel_in, ch_out=channel_out)
        self.block2 = Conv2(ch_in=channel_out, ch_out=channel_out)
        self.block3 = Conv2(ch_in=channel_out, ch_out=channel_out)
        self.block4 = Conv2(ch_in=channel_out, ch_out=channel_out)
        self.block5 = Conv2(ch_in=channel_out, ch_out=channel_out)
        self.pool = patchmerging(channel_out)
    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.pool(y)
        return y

# --- Transformer 相关模块 ---

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, axis, fn):
        super().__init__()
        self.norm = nn.LayerNorm(axis)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 深度可分离卷积
class DWC_conv(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.h = h
        self.w = w
        self.conv = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c)
    def forward(self, x):
        """
        :param x: [B, P, N, C]
        """
        B, P, N, C = x.shape
        # [B, P, N, C] -> [B, C, P*N] -> [B, C, H, W]
        ts = x.permute(0, 3, 1, 2).contiguous().view(B, C, -1)
        
        # Dynamically calculate H and W
        H_W_ = P * N
        H = W = int(math.sqrt(H_W_))

        ts = ts.view(B, C, H, W)
        ts = self.conv(ts)
        # [B, C, H, W] -> [B, C, P*N] -> [B, P, N, C]
        ts = ts.view(B, C, -1).permute(0, 2, 1).contiguous()
        ts = ts.view(B, P, N, C)
        return ts

class FeedForward(nn.Module):
    def __init__(self, axis, hidden_axis, h, w, dropout=0.):
        super().__init__()
        self.h = h
        self.w = w
        self.fc1 = nn.Linear(axis, hidden_axis)
        self.DW = DWC_conv(hidden_axis, self.h, self.w)
        self.norm = nn.LayerNorm(hidden_axis)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_axis, axis)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.DW(x1)
        x2 = x1 + x2
        x2 = self.norm(x2)
        x2 = self.act(x2)
        x2 = self.fc2(x2)
        return x2

class Attention(nn.Module):
    def __init__(self, axis, heads=8, axis_head=64, dropout=0.):
        super().__init__()
        inner_axis = axis_head * heads
        project_out = not (heads == 1 and axis_head == axis)

        self.heads = heads
        self.scale = axis_head ** -0.5
        self.to_qkv = nn.Linear(axis, inner_axis * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_axis, axis),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv

        b, p, n, hd = q.shape

        # Reshape for multi-head attention
        # [b, p, n, heads * axis_head] -> [b, p, n, heads, axis_head] -> [b, p, heads, n, axis_head]
        q = q.view(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)
        k = k.view(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)
        v = v.view(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)

        # Dots: [b, p, heads, n, n]
        dots = torch.matmul(q, k.permute(0, 1, 2, 4, 3)) * self.scale
        attn = F.softmax(dots, dim=-1)

        # Out: [b, p, heads, n, axis_head]
        out = torch.matmul(attn, v)
        # Back to [b, p, n, heads*axis_head]
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(b, p, n, -1)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, axis, depth, heads, axis_head, mlp_axis, h, w, dropout=0.):
        super().__init__()
        self.h = h
        self.w = w
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(axis, Attention(axis, heads, axis_head, dropout)),
                PreNorm(axis, FeedForward(axis, mlp_axis, self.h, self.w, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, axis, depth, channel, kernel_size, patch_size, mlp_axis, h, w, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.h = h
        self.w = w

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, axis)

        self.transformer = Transformer(axis, depth, 1, 32, mlp_axis, self.h, self.w, dropout)

        self.conv3 = conv_1x1_bn(axis, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # 局部表示
        x = self.conv1(x)
        x = self.conv2(x)

        # 全局表示
        n, c, h, w = x.shape

        # Unfold: [N, C, H, W] -> [N, P, PatchArea, C]
        # P = patches count, PatchArea = ph*pw
        x = x.permute(0, 2, 3, 1).contiguous().view(n, self.ph * self.pw, -1, c)

        x = self.transformer(x)

        # Fold: [N, P, PatchArea, C] -> [N, C, H, W]
        x = x.view(n, h, w, c).permute(0, 3, 1, 2)

        # 融合
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class Convblock1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Convblock1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        y = self.conv(x)
        return y

class TransformerEncoder(nn.Module):
    def __init__(self, image_size, axiss, channels, mlp_axis, channel1, h, w, kernel_size=3, patch_size=(2, 2)):
        super(TransformerEncoder, self).__init__()
        self.h = h
        self.w = w
        ih, iw = image_size
        ph, pw = patch_size
        # assert ih % ph == 0 and iw % pw == 0

        self.mv1 = MobileViTBlock(axiss, 2, channels, kernel_size, patch_size, mlp_axis, self.h, self.w)
        self.conv1 = Convblock1(channel1, channels)
        self.pool = patchmerging(channels)

    def forward(self, x):
        x = self.conv1(x)
        y = self.mv1(x)
        y = self.pool(y)
        return y

# --- 其他功能模块 ---

class conv11_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv11_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class conv33_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv33_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return self.conv(x)

class AF_block(nn.Module):
    def __init__(self, ch_in, ch_out, ch_middle):
        super(AF_block, self).__init__()
        self.conv = conv11_block(ch_in * 2, ch_in)
        self.act = nn.Sigmoid()
        self.conv2 = conv33_block(ch_in, 1)

    def forward(self, a, b):
        c = torch.cat([a, b], dim=1)
        c = self.conv(c)
        c1 = self.conv2(c)
        c2 = self.act(c1)
        out = c2 * a + (1 - c2) * b
        return out

class CLF_block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(CLF_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=1, stride=1, padding=0)
        self.act = nn.Softmax(dim=-1)

    def forward(self, a, b):
        z = torch.cat([a, b], dim=1)
        z = self.conv1(z)
        B, C, H, W = z.shape
        q = self.conv2(z)
        k = self.conv3(z)
        v = self.conv4(z)

        q = q.view(B, C, -1)  # b c hw
        k = k.view(B, C, -1)  # b c hw
        v = v.view(B, C, -1)  # b c hw

        k = k.permute(0, 2, 1)  # b hw c
        qk = torch.matmul(q, k)  # b c c
        qk = self.act(qk)

        out = torch.matmul(qk, v)  # b c hw
        out = out.view(B, C, H, W)
        return out

class MFE_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MFE_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in * 2, ch_in, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        # 注意：这里 CLF 输入是 u和v, u是ch_out, v是ch_out, concat后是 ch_out*2
        # 原代码写的是 channel_in=1024, channel_out=512，假设 ch_in=512
        self.clf1 = CLF_block(channel_in=ch_out*2, channel_out=ch_out)
        self.clf2 = CLF_block(channel_in=ch_out*2, channel_out=ch_out)

    def forward(self, a, b):
        x = torch.cat([a, b], dim=1)
        x = self.conv(x)
        u = self.conv1(x)
        v = self.conv2(x)
        w = self.conv3(x)
        y = self.clf1(a=u, b=v)
        z = self.clf2(a=y, b=w)
        out = z + x
        return out

class ALFEblock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ALFEblock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in * 2, ch_in, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=1)
        self.conv6 = nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=1)
        self.act = nn.Softmax(dim=-1)
        self.pool1 = patchmerging(ch_in)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = x
        c1 = self.pool1(x)

        B, C, H, W = c1.shape

        q = self.conv2(c1)
        k = self.conv3(c1)
        v = self.conv4(c1)

        q1 = q.view(B, C, -1).permute(0, 2, 1)  # B HW C
        k1 = k.view(B, C, -1)  # B C HW
        v = v.view(B, C, -1)

        qk1 = torch.matmul(q1, k1)  # B HW HW
        qk1 = self.act(qk1)
        qk1 = qk1.permute(0, 2, 1)  # B HW HW
        out1 = torch.matmul(v, qk1)
        out1 = out1.view(B, C, H, W)

        q2 = self.conv5(c1)
        k2 = self.conv6(c1)
        q2 = q2.view(B, C, -1)
        k2 = k2.view(B, C, -1).permute(0, 2, 1)
        qk2 = torch.matmul(q2, k2)
        qk2 = self.act(qk2)
        out2 = torch.matmul(qk2, v)
        out2 = out2.view(B, C, H, W)

        out = torch.cat([out1, out2], dim=1)
        out = self.conv1(out)
        out = self.up(out)
        out = out + x1

        return out

class dsconv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(dsconv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=ch_in),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, groups=ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ER_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ER_block, self).__init__()
        self.conv = dsconv_block(ch_in, ch_out)

    def forward(self, x):
        y = self.conv(x)
        y = y + x
        return y

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.up(x)

# 边缘检测卷积模块
# 原代码中使用 fluid.dygraph.to_variable 和 numpy 初始化
# PyTorch 中建议将其作为 nn.Module 的一部分，并将核注册为 buffer
class SobelEdgeDetector(nn.Module):
    def __init__(self):
        super(SobelEdgeDetector, self).__init__()
        # Sobel X
        sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
        self.register_kernel(sobel_kernel, 'k1')

        # Sobel Y
        sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
        self.register_kernel(sobel_kernel1, 'k2')

        # 45度对角线
        sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
        self.register_kernel(sobel_kernel2, 'k3')

        # 135度对角线
        sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
        self.register_kernel(sobel_kernel3, 'k4')

    def register_kernel(self, k, name):
        k = k.reshape((1, 1, 3, 3))
        k = np.repeat(k, 3, axis=1) # in_channels=3
        k = np.repeat(k, 3, axis=0) # out_channels=3
        self.register_buffer(name, torch.from_numpy(k))

    def forward(self, x):
        edge1 = torch.pow(F.conv2d(x, self.k1, padding=1), 2)
        edge2 = torch.pow(F.conv2d(x, self.k2, padding=1), 2)
        edge3 = torch.pow(F.conv2d(x, self.k3, padding=1), 2)
        edge4 = torch.pow(F.conv2d(x, self.k4, padding=1), 2)

        sobel_out = edge1 + edge2 + edge3 + edge4
        sobel_out = torch.sqrt(sobel_out + 1e-6) # Add epsilon for stability
        return sobel_out

# 主网络模型
class Three_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, image_size=256):
        super(Three_Net, self).__init__()

        # Dynamically calculate feature map sizes
        h, w = image_size, image_size
        h2, w2 = h // 2, w // 2
        h3, w3 = h2 // 2, w2 // 2
        h4, w4 = h3 // 2, w3 // 2
        h5, w5 = h4 // 2, w4 // 2 # For TransformerEncoder's image_size parameter

        # --- 卷积分支 (边缘特征) ---
        self.edge_detector = SobelEdgeDetector()
        self.block11 = conv_block(ch_in=3, ch_out=64)
        self.pool = patchmerging(ch_in=64)
        self.block12 = encoder_block2(channel_in=64, channel_out=128)
        self.block13 = encoder_block3(channel_in=128, channel_out=256)
        self.block14 = encoder_block4(channel_in=256, channel_out=512)

        # --- Transformer 分支 (上下文特征) ---
        self.block21 = TransformerEncoder(image_size=(h, w), axiss=96, channels=64, channel1=3, patch_size=(8, 8), mlp_axis=96*2, h=h, w=w)
        self.block22 = TransformerEncoder(image_size=(h2, w2), axiss=192, channels=128, channel1=64, patch_size=(8, 8), mlp_axis=192*2, h=h2, w=w2)
        self.block23 = TransformerEncoder(image_size=(h3, w3), axiss=384, channels=256, channel1=128, patch_size=(2, 2), mlp_axis=384*2, h=h3, w=w3)
        self.block24 = TransformerEncoder(image_size=(h4, w4), axiss=768, channels=512, channel1=256, patch_size=(2, 2), mlp_axis=768*2, h=h4, w=w4)

        # --- 特征融合与增强模块 ---
        self.AF1 = AF_block(ch_in=64, ch_out=64, ch_middle=8)
        self.AF2 = AF_block(ch_in=128, ch_out=128, ch_middle=16)
        self.AF3 = AF_block(ch_in=256, ch_out=256, ch_middle=32)
        self.AF4 = AF_block(ch_in=512, ch_out=512, ch_middle=64)

        self.ALFE64 = ALFEblock(ch_in=64, ch_out=64)
        self.ALFE128 = ALFEblock(ch_in=128, ch_out=128)
        self.ALFE256 = ALFEblock(ch_in=256, ch_out=256)
        self.ALFE512 = ALFEblock(ch_in=512, ch_out=512)

        self.ER64 = ER_block(ch_in=64, ch_out=64)
        self.ER128 = ER_block(ch_in=128, ch_out=128)
        self.ER256 = ER_block(ch_in=256, ch_out=256)
        self.ER512 = ER_block(ch_in=512, ch_out=512)

        self.Deepest = MFE_block(ch_in=512, ch_out=512)

        # --- 解码器 ---
        self.Conv11 = conv_block(1024, 512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Up1 = up_conv(ch_in=64, ch_out=32)
        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        :param x: 输入图像张量，形状 [B, 3, H, W]。
        :return: 输出分割图张量，形状 [B, output_ch, H, W]。
        """
        # --- Transformer 分支 ---
        t1 = self.block21(x)
        t2 = self.block22(t1)
        t3 = self.block23(t2)
        t4 = self.block24(t3)

        # --- 卷积分支 ---
        x_edge = self.edge_detector(x)
        x1 = self.block11(x_edge)
        x1_pool = self.pool(x1)

        # --- 编码器与特征融合 ---
        out1 = self.ER64(self.ALFE64(self.AF1(a=x1_pool, b=t1)))
        x2 = self.block12(x1_pool)
        out2 = self.ER128(self.ALFE128(self.AF2(a=x2, b=t2)))
        x3 = self.block13(x2)
        out3 = self.ER256(self.ALFE256(self.AF3(a=x3, b=t3)))
        x4 = self.block14(x3)
        out4 = self.ER512(self.ALFE512(self.AF4(a=x4, b=t4)))

        # --- 最深层 ---
        out5 = self.Deepest(a=x4, b=t4)
        d5 = torch.cat([out5, out4], dim=1)
        d5 = self.Conv11(d5)

        # --- 解码器 ---
        d4 = self.Up4(d5)
        d4 = torch.cat([d4, out3], dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat([d3, out2], dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat([d2, out1], dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)

        d1 = self.Conv_1x1(d1)

        return d1

# --- 模型测试 ---
if __name__ == "__main__":
    IMAGE_SIZE = 256
    num_classes = 1
    model = Three_Net(img_ch=3, output_ch=num_classes, image_size=IMAGE_SIZE)

    # 创建 Dummy Input [B, C, H, W]
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    # 检查是否有 GPU
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()

    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

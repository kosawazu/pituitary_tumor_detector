import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_channels = in_channels
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, stride=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 入力テンソル`x`のデバイスに合わせる
        device = x.device
        self.fc = self.fc.to(device)

        # チャネル数が異なる場合、再定義
        if x.size(1) != self.in_channels:
            in_channels = x.size(1)
            self.fc = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, stride=1).to(device),
                nn.ReLU(),
                nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, stride=1).to(device)
            )
            self.in_channels = in_channels

        avg_out = self.fc(self.avg_pool(x))
        return self.sigmoid(avg_out)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # query, key, valueの変換
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # (B, W*H, C//8)
        key = self.key(x).view(batch_size, -1, width * height)  # (B, C//8, W*H)
        value = self.value(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # (B, W*H, C)

        # Attentionの計算
        attention = torch.bmm(query, key)  # (B, W*H, W*H)
        attention = F.softmax(attention, dim=-1)

        # ValueとAttentionをかけ合わせる
        out = torch.bmm(attention, value).view(batch_size, C, width, height)  # (B, C, W, H)
        
        # 元の入力xにgamma * outを加算
        out = self.gamma * out + x
        return out







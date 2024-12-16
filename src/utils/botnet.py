import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# 相対位置エンコーディングモジュール
class RelativePosEnc(nn.Module):
    """特徴マップの相対位置エンコーディングを実装するモジュール"""
    def forward(self, q):
        if not hasattr(self, 'pos_h'):  # 相対位置埋め込みを初期化
            c, h, w = q.shape[-3:]
            self.pos_h = nn.Parameter(torch.zeros(2 * h - 1, c, device=q.device))
            self.pos_w = nn.Parameter(torch.zeros(2 * w - 1, c, device=q.device))

        # 相対位置エンコーディングを計算
        rel_h = self.pos_h @ q.movedim(4, 2)
        rel_w = self.pos_w @ q.movedim(3, 2)

        # 絶対位置エンコーディングへ変換
        rel_h = self.rel_to_abs(rel_h).movedim(2, 4)
        rel_w = self.rel_to_abs(rel_w).movedim(2, 3)

        # 高さと幅のエンコーディングを結合
        pos_enc = rel_h[:, :, :, None] + rel_w[:, :, None, :]
        pos_enc = pos_enc.flatten(-2).flatten(2, 3)

        return pos_enc

    @staticmethod
    def rel_to_abs(x):
        """相対位置エンコーディングを絶対位置エンコーディングに変換"""
        shape = x.shape
        length = shape[-1]
        x = F.pad(x, (0, 0, 0, 1))  # パディングを追加
        x = x.flatten(-2)
        x = F.pad(x, (0, length-1))
        x = x.view(*shape[:-1], length+1)
        x = x[..., length-1:, :length]
        return x


# 自己注意機構 (Multi-Head Self-Attention)
class SelfAttention2d(nn.Module):
    """2次元の自己注意モジュール"""
    def __init__(self, in_channels, out_channels, q_channels, v_channels, heads, pos_enc, p_drop=0.):
        super().__init__()
        self.heads = heads
        self.q_channels = q_channels
        self.scale = q_channels ** -0.5
        self.to_pos_enc = pos_enc()  # 相対位置エンコーディング
        self.to_keys = nn.Conv2d(in_channels, q_channels * heads, 1)
        self.to_queries = nn.Conv2d(in_channels, q_channels * heads, 1)
        self.to_values = nn.Conv2d(in_channels, v_channels * heads, 1)
        self.unifyheads = nn.Conv2d(v_channels * heads, out_channels, 1)
        self.attn_drop = nn.Dropout(p_drop)
        self.resid_drop = nn.Dropout(p_drop)

    def forward(self, x):
        bs, _, h, w = x.shape
        # キー、クエリ、バリューを計算
        keys = self.to_keys(x).view(bs, self.heads, self.q_channels, h * w)
        queries = self.to_queries(x).view(bs, self.heads, self.q_channels, h * w)
        values = self.to_values(x).view(bs, self.heads, -1, h * w)
        
        # 相対位置エンコーディングを適用
        pos_enc = self.to_pos_enc(queries.view(bs, self.heads, self.q_channels, h, w))
        
        # 注意スコアを計算
        att = torch.matmul(queries.transpose(-2, -1), keys) + pos_enc
        att = F.softmax(att * self.scale, dim=-1)
        att = self.attn_drop(att)
        
        # 出力を計算
        out = torch.matmul(att, values.transpose(-2, -1))
        out = out.view(bs, -1, h, w)
        out = self.unifyheads(out)
        out = self.resid_drop(out)
        return out


# 自己注意ブロック
class AttentionBlock(nn.Sequential):
    """Self-attentionのブロック化"""
    def __init__(self, channels, heads=4, p_drop=0.):
        q_channels = channels // heads
        super().__init__(
            SelfAttention2d(channels, channels, q_channels, q_channels, heads, RelativePosEnc, p_drop),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )


# 畳み込みブロック
class ConvBlock(nn.Sequential):
    """ボトルネック構造に用いる畳み込みブロック"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=True):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


# BoTNetの残差ブロック
class BoTResidual(nn.Sequential):
    """BoTNet用の残差ブロック"""
    def __init__(self, in_channels, out_channels, expansion=4, heads=4, p_drop=0.):
        bottl_channels = out_channels // expansion
        super().__init__(
            ConvBlock(in_channels, bottl_channels, kernel_size=1, stride=1),
            AttentionBlock(bottl_channels, heads, p_drop),
            ConvBlock(bottl_channels, out_channels, kernel_size=1, act=False)
        )


# 残差接続を含むモジュール
class ResidualBlock(nn.Module):
    """残差接続を適用するモジュール"""
    def __init__(self, in_channels, out_channels, residual):
        super().__init__()
        self.shortcut = self.get_shortcut(in_channels, out_channels)
        self.residual = residual(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))  # 学習可能スケールパラメータ

    def forward(self, x):
        out = self.shortcut(x) + self.gamma * self.residual(x)
        return self.relu(out)

    def get_shortcut(self, in_channels, out_channels):
        """入力チャンネル数と出力チャンネル数が異なる場合、1x1の畳み込みで調整"""
        if in_channels != out_channels:
            return ConvBlock(in_channels, out_channels, kernel_size=1, act=False)
        else:
            return nn.Identity()


# ResNet101をカスタマイズしてBoTNetを構築
class CustomResNet101(nn.Module):
    """BoTNetを使用するResNet101ベースのエンコーダー"""
    def __init__(self):
        super(CustomResNet101, self).__init__()
        resnet = models.resnet101(pretrained=True)  # 事前学習済みモデルをロード
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = self._make_layer(512, 1024, 23, BoTResidual, 4)  # layer3をBoTNetに置き換え
        self.layer4 = resnet.layer4

    def _make_layer(self, in_channels, out_channels, blocks, residual_block, heads):
        """カスタムレイヤーを構築"""
        layers = [
            ResidualBlock(
                in_channels,
                out_channels,
                lambda in_c, out_c: residual_block(in_c, out_c, heads=heads)
            )
        ]
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    lambda in_c, out_c: residual_block(in_c, out_c, heads=heads)
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        """エンコーダーの順伝播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# デコーダー部分
class SegmentationDecoder(nn.Module):
    """セグメンテーションのためのデコーダー"""
    def __init__(self, num_classes):
        super(SegmentationDecoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x, input_size):
        """デコーダーの順伝播"""
        x = self.up1(x)
        x = F.relu(x)
        x = self.up2(x)
        x = F.relu(x)
        x = self.up3(x)
        x = F.relu(x)
        x = self.up4(x)
        x = F.relu(x)
        x = self.final_conv(x)
        # 入力サイズにリサイズ
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x


# BoTNetを使用したセグメンテーションモデル
class fcn_bot_resnet101(nn.Module):
    """エンコーダー + デコーダーの統合モデル"""
    def __init__(self, num_classes):
        super(fcn_bot_resnet101, self).__init__()
        self.encoder = CustomResNet101()
        self.decoder = SegmentationDecoder(num_classes)

    def forward(self, x):
        input_size = x.shape[-2:]  # 入力画像の解像度を保存
        x = self.encoder(x)        # エンコーダーの出力
        x = self.decoder(x, input_size)  # デコーダーの出力
        return x
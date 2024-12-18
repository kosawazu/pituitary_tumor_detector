import torch
import torch.nn as nn
import torchvision.models as models

class ViTSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(ViTSegmentation, self).__init__()
        
        # ViT-Base モデルをロード（事前学習済み）
        self.vit = models.vit_b_16(pretrained=True)
        
        # ViTの出力次元（通常は768）
        vit_output_dim = self.vit.hidden_dim
        
        # 中間特徴マップのサイズ
        self.patch_size = self.vit.patch_size
        self.num_patches = (224 // self.patch_size) ** 2
        
        # マルチスケール特徴を処理するための層
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(vit_output_dim, 256, kernel_size=1)
            for _ in range(3)  # 3つの異なるスケールを使用
        ])
        
        # デコーダー部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 * 3, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 入力サイズを保存
        original_size = x.shape[-2:]
        
        # ViTの中間層の出力を取得
        features = []
        for i, block in enumerate(self.vit.encoder.layers):
            x = block(x)
            if i in [3, 7, 11]:  # 異なる深さの特徴を選択
                features.append(x)
        
        # 特徴を2D形式に変換し、処理
        processed_features = []
        for i, feature in enumerate(features):
            feature = feature[:, 1:, :].transpose(1, 2).view(-1, self.vit.hidden_dim, int(self.num_patches**0.5), int(self.num_patches**0.5))
            processed_features.append(self.multi_scale_conv[i](feature))
        
        # 特徴を結合
        x = torch.cat(processed_features, dim=1)
        
        # デコード
        x = self.decoder(x)
        
        # 元のサイズにリサイズ
        x = nn.functional.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x
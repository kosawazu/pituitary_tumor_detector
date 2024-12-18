import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class vit_segmentation(nn.Module):
    def __init__(self, num_classes):
        super(vit_segmentation, self).__init__()
        
        # ViT-Base モデルをロード（事前学習済み）
        self.vit = models.vit_b_16(pretrained=True)
        
        # ViTの出力次元（通常は768）
        vit_output_dim = self.vit.hidden_dim
        
        # パッチサイズとパッチの数を取得
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
        
        # パッチ埋め込みを適用
        if hasattr(self.vit, 'patch_embed'):
            x = self.vit.patch_embed(x)
        elif hasattr(self.vit, 'conv_proj'):
            x = self.vit.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)
        else:
            raise AttributeError("ViT model structure is not recognized")
        
        # クラストークンを追加
        if hasattr(self.vit, 'class_token'):
            cls_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        else:
            cls_token = self.vit.transformer.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 位置埋め込みを追加
        if hasattr(self.vit, 'pos_embed'):
            pos_embed = self.vit.pos_embed
        elif hasattr(self.vit, 'encoder') and hasattr(self.vit.encoder, 'pos_embedding'):
            pos_embed = self.vit.encoder.pos_embedding
        else:
            raise AttributeError("Position embedding not found in the ViT model")
        x = x + pos_embed
        
        # ViTの中間層の出力を取得
        features = []
        if hasattr(self.vit, 'encoder') and hasattr(self.vit.encoder, 'layers'):
            for i, block in enumerate(self.vit.encoder.layers):
                x = block(x)
                if i in [3, 7, 11]:  # 異なる深さの特徴を選択
                    features.append(x)
        elif hasattr(self.vit, 'transformer') and hasattr(self.vit.transformer, 'encoder'):
            for i, block in enumerate(self.vit.transformer.encoder.layers):
                x = block(x)
                if i in [3, 7, 11]:  # 異なる深さの特徴を選択
                    features.append(x)
        else:
            raise AttributeError("ViT encoder structure not recognized")
        
        # 特徴を2D形式に変換し、処理
        processed_features = []
        for i, feature in enumerate(features):
            feature = feature[:, 1:, :].transpose(1, 2).reshape(feature.shape[0], -1, int(np.sqrt(feature.shape[1]-1)), int(np.sqrt(feature.shape[1]-1)))
            processed_features.append(self.multi_scale_conv[i](feature))
        
        # 特徴を結合
        x = torch.cat(processed_features, dim=1)
        
        # デコード
        x = self.decoder(x)
        
        # 元のサイズにリサイズ
        x = nn.functional.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x
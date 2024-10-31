import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

def setup_device(
    model: nn.Module,
    model_path: Path = None
) -> torch.nn.Module:
    """モデルをデバイス（GPU/CPU）に設定し、必要に応じてマルチGPUモードに切り替えます。"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            logger.info(f"Let's use {device_count} GPUs!")
            model = torch.nn.DataParallel(model)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.info("CUDA is not available. Using CPU.")
    model.to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return device, model

def setup_fcn_model(
    model_name: str,
    num_classes: int =3
) -> nn.Module:
    # モデルのロード（事前学習済みのモデルをファインチューニング）
    model_dict = {
    "fcn_resnet50":models.segmentation.fcn_resnet50(pretrained=True),
    "fcn_resnet101":models.segmentation.fcn_resnet101(pretrained=True),
    "fcn_vgg16": models.vgg16(pretrained=True),
    "fcn_vgg19": models.vgg19(pretrained=True),
    "deeplabv3_resnet101":models.segmentation.deeplabv3_resnet101(pretrained=True)
    }
    if model_name not in model_dict:
        raise ValueError(f"Invalid model_name '{model_name}'.")
    model = model_dict[model_name]
    model = tune_model(model, model_name, num_classes)
    return model

def tune_model(
    model: nn.Module,
    model_name: str,
    num_classes: int
) -> nn.Module:
    # クラス数を紙袋検出用に調整（背景+紙袋 = 2クラス）
    if model_name == "deeplabv3_resnet101":
        logger.info("モデルです")
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))  # num_classesに出力クラス数を設定
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif "resnet" in model_name:
        # ResNetの場合（512チャンネル）
        model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif "vgg" in model_name:
        # VGGの特徴抽出部を取得
        features = list(model.features.children())
        
        # VGGの全結合層部分を削除し、畳み込み層に置き換え
        classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),  # FCN部分の追加
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, num_classes, kernel_size=1)  # 出力層
        )

        # VGGモデルをセグメンテーションに適した形で再構築
        model = nn.Sequential(
            *features,
            classifier
        )

        # 出力を辞書形式で返すように調整
        class VGG_FCN(nn.Module):
            def __init__(self, vgg_model):
                super(VGG_FCN, self).__init__()
                self.vgg_model = vgg_model

            def forward(self, x):
                # VGGモデルの出力をセグメンテーション形式に整える
                x = self.vgg_model(x)
                return {"out": x}  # 辞書形式で出力

        # ラップして辞書形式で返すモデルに変換
        model = VGG_FCN(model)

    return model
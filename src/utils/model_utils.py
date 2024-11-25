import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from utils.attention_layers import (
    SelfAttention,
    ChannelAttention
)

logger = logging.getLogger(__name__)

def setup_device(
    model: nn.Module,
    model_path: Path = None
) -> nn.Module:
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
    attention_mode: str,
    num_classes: int =3, 
) -> nn.Module:
    # モデルのロード（事前学習済みのモデルをファインチューニング）
    model_dict = {
    "fcn_resnet50":models.segmentation.fcn_resnet50(pretrained=True),
    "fcn_resnet101":models.segmentation.fcn_resnet101(pretrained=True),
    "deeplabv3_resnet101":models.segmentation.deeplabv3_resnet101(pretrained=True)
    }
    if model_name not in model_dict:
        raise ValueError(f"Invalid model_name '{model_name}'.")
    model = model_dict[model_name]
    model = tune_model(model, model_name, attention_mode, num_classes)
    return model

def tune_model(
    model: nn.Module,
    model_name: str,
    attention_mode: str,
    num_classes: int =3
) -> nn.Module:
    self_attention = SelfAttention(2048)  # layer4の出力チャンネル数は2048
    channel_attention = ChannelAttention(2048)
    if model_name == "deeplabv3_resnet101":
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))  # num_classesに出力クラス数を設定
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif "resnet" in model_name:
        # ResNetの場合（512チャンネル）
        model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    # アテンションモードに基づいてbackbone.layer4を設定
    original_layer4 = model.backbone.layer4
    # Self-Attentionとchannel_atttentionをlayer4に追加
    # layer4に注意層を追加する
    if attention_mode == "none":
        # アテンションを適用しない
        pass
    elif attention_mode == "self_attention":
        model.backbone.layer4 = nn.Sequential(original_layer4, self_attention)
    elif attention_mode == "channel_attention":
        model.backbone.layer4 = nn.Sequential(channel_attention, original_layer4)
    elif attention_mode == "both":
        model.backbone.layer4 = nn.Sequential(channel_attention, original_layer4, self_attention)
    else:
        raise ValueError(f"Invalid attention mode: {attention_mode}")
    return model
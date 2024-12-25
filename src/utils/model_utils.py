import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

#自作モジュールのimport
from utils.attention_layers import (
    SelfAttention,
    ChannelAttention
)
from utils.botnet import fcn_bot_resnet101
from utils.vit import vit_segmentation
from transformers import SegformerForSemanticSegmentation

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
        if torch.cuda.is_available():
            if device_count > 1:
                model.load_state_dict(torch.load(model_path))
            if device_count == 1:
                loaded_state_dict = torch.load(model_path)
                new_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
                model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
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
    "deeplabv3_resnet101":models.segmentation.deeplabv3_resnet101(pretrained=True),
    "fcn_bot_resnet101":fcn_bot_resnet101(num_classes),
    "segformer_b0": SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512"),
    "segformer_b4": SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
    }
    if model_name not in model_dict:
        raise ValueError(f"Invalid model_name '{model_name}'.")
    model = model_dict[model_name]
    if "segformer" in model_name:
        # SegFormerモデルのクラス数を調整
        if num_classes != model.decode_head.classifier.out_channels:
            model.decode_head.classifier = nn.Conv2d(
                model.decode_head.classifier.in_channels,
                num_classes,
                kernel_size=1
            )
    model = tune_model(model, model_name, attention_mode, num_classes)
    return model

def tune_model(
    model: nn.Module,
    model_name: str,
    attention_mode: str,
    num_classes: int = 3
) -> nn.Module:
    
    if model_name == "fcn_bot_resnet101":
        # BoTNetのlayer3以降を学習可能に設定
        for param in model.encoder.layer1.parameters():
            param.requires_grad = False
        for param in model.encoder.layer2.parameters():
            param.requires_grad = False
        
        # layer3とlayer4は学習可能
        for param in model.encoder.layer3.parameters():
            param.requires_grad = True
        for param in model.encoder.layer4.parameters():
            param.requires_grad = True
    elif model_name == "segformer_b0" or model_name == "segformer_b4":
        pass
    else:
        self_attention = SelfAttention(2048)  # layer4の出力チャンネル数は2048
        channel_attention = ChannelAttention(2048)

        if model_name == "deeplabv3_resnet101":
            model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        elif "resnet" in model_name:
            model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # アテンションモードに基づいてbackbone.layer4を設定
        original_layer4 = model.backbone.layer4

        if attention_mode == "none":
            model.backbone.layer4 = nn.Sequential(original_layer4)
        elif attention_mode == "self_attention":
            model.backbone.layer4 = nn.Sequential(original_layer4, self_attention)
        elif attention_mode == "channel_attention":
            model.backbone.layer4 = nn.Sequential(channel_attention, original_layer4)
        elif attention_mode == "both":
            model.backbone.layer4 = nn.Sequential(channel_attention, original_layer4, self_attention)
        else:
            raise ValueError(f"Invalid attention mode: {attention_mode}")

    return model

# def tune_model(
#     model: nn.Module,
#     model_name: str,
#     attention_mode: str,
#     num_classes: int
# ) -> nn.Module:
#     # クラス数を紙袋検出用に調整（背景+紙袋 = 2クラス）
#     if model_name == "deeplabv3_resnet101":
#         attention_layer = SelfAttention(2048)  # layer4の出力チャンネル数は2048
#         channel_attention_layer = ChannelAttention(2048)
#         model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))  # num_classesに出力クラス数を設定
#         model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
#         model.backbone.layer4 = nn.Sequential(
#             # channel_attention_layer,  # まずはChannelAttentionを適用
#             model.backbone.layer4,    # その後に既存のlayer4
#             # attention_layer           # そしてSelfAttention
#         )
#     elif "resnet" in model_name:
#         # 注意層を初期化
#         attention_layer = SelfAttention(2048)
#         channel_attention_layer = ChannelAttention(2048)
#         # ResNetの場合（512チャンネル）
#         model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
#         # Self-Attentionとchannel_atttentionをlayer4に追加
#         # layer4に注意層を追加する
#         model.backbone.layer4 = nn.Sequential(
#             # channel_attention_layer,  # ChannelAttentionをまず適用
#             model.backbone.layer4,    # 次に既存のlayer4
#             # attention_layer           # 最後にSelfAttentionを追加
#         )
#     return model
import logging
from pathlib import Path
import sys
import os
import torch
import torch.nn as nn
from torchvision import models
# 自作モジュールのインポート
try:
    from utils.botnet import fcn_bot_resnet101
    from utils.vit import vit_segmentation
except ImportError as e:
    raise ImportError(f"Failed to import custom modules: {e}")

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # コンソール出力
        logging.FileHandler("debug.log")  # ファイル出力
    ]
)
logger = logging.getLogger(__name__)

def setup_device(
    model: nn.Module, 
    model_path: Path = None
) -> nn.Module:
    """
    モデルをデバイス（GPU/CPU）に設定し、必要に応じてマルチGPUモードに切り替えます。
    """
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
    
    try:
        logger.info(f"Loading model from {model_path}...")
        if device_count > 1:
            # マルチGPUの場合はmap_locationを指定しない
            logger.info(f"GPUの数が{device_count}のためそのまま読み込みます。")
            loaded_state_dict = torch.load(model_path)
            model.load_state_dict(loaded_state_dict)
        else:
            # シングルGPUまたはCPUの場合は明示的にデバイスを指定
            loaded_state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(loaded_state_dict, strict=False)
            logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise
    return device, model


def setup_model(model_name: str, num_classes: int = 3) -> nn.Module:
    """
    モデルのロード（事前学習済みのモデルをファインチューニング）
    """
    model_dict = {
        "fcn_resnet50": models.segmentation.fcn_resnet50(weights=None),
        "fcn_resnet101": models.segmentation.fcn_resnet101(weights=None),
        "deeplabv3_resnet101": models.segmentation.deeplabv3_resnet101(weights=None, aux_loss=True),
        "fcn_bot_resnet101": fcn_bot_resnet101(num_classes),
        "vit_b_16_segmentation": vit_segmentation(num_classes)
    }
    if model_name not in model_dict:
        logger.error(f"Invalid model_name '{model_name}'.")
        raise ValueError(f"Invalid model_name '{model_name}'.")
    
    model = model_dict[model_name]
    model = tune_model(model, model_name, num_classes)
    return model

def tune_model(model: nn.Module, model_name: str, num_classes: int = 3) -> nn.Module:
    """
    モデルの構造を調整
    """
    if model_name == "fcn_bot_resnet101":
        # BoTNet の特定層の学習設定
        for param in model.encoder.layer1.parameters():
            param.requires_grad = False
        for param in model.encoder.layer2.parameters():
            param.requires_grad = False
        for param in model.encoder.layer3.parameters():
            param.requires_grad = True
        for param in model.encoder.layer4.parameters():
            param.requires_grad = True
    elif model_name == "vit_b_16_segmentation":
        pass
    else:
        if model_name == "deeplabv3_resnet101":
            model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        elif "resnet" in model_name:
            model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Attention の設定
        original_layer4 = model.backbone.layer4
        model.backbone.layer4 = nn.Sequential(original_layer4)
    return model

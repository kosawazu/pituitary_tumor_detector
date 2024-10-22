from pathlib import Path
import torch
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
from torch import Tensor

from PIL import Image
from typing import Tuple

#使用例
def main():
    model_path = Path("model", "best_model_segment.pth")
    image_path = Path("data", "a.png")
    model = setup_model(model_path, num_classes=3)
    device, model = setup_device(model)
    img = Image.open(image_path).convert('RGB')
    segment_image = image_segment(img, device, model)

    return segment_image

def get_transform():
    return transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#デバイスを定義する関数
def setup_device(
    model: nn.Module
) -> Tuple[torch.device, torch.nn.Module]:
    """モデルをデバイス（GPU/CPU）に設定し、必要に応じてマルチGPUモードに切り替えます。"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Let's use {device_count} GPUs!")
            model = torch.nn.DataParallel(model)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    model.to(device)
    return device, model

#モデルをロードする関数
def setup_model(
    model_path: Path, 
    num_classes: int =3
) -> nn.Module:
    # モデルのロード（事前学習済みのモデルをファインチューニング）
    model = models.segmentation.fcn_resnet50(pretrained=True)

    # クラス数を紙袋検出用に調整（背景+紙袋 = 2クラス）
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(model_path))
    return model


#セグメントを行う処理
def image_segment(
    img: Image.Image,
    device: torch.device,
    model: nn.Module
) -> np.ndarray:
    transform = get_transform()
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0).to(device)  # バッチ次元を追加
    with torch.no_grad():
        output = model(input_batch)['out']  # FCNの出力
        # 各ピクセルに最も確率の高いクラスを割り当てる
        output_predictions = output.argmax(1).squeeze().cpu().numpy()
    return output_predictions

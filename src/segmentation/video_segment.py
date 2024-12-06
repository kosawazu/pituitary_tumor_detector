import os
import argparse
import logging
from pathlib import Path
import sys
import cv2
import torch
from torchvision import models, transforms
import torch.optim as optim

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Callable

sys.path.append("../")
# 自作モジュール
from segmentation import (
    transform
)

from utils.model_utils import (
    setup_fcn_model,
    setup_device
)

from segment_utils.dataset_utils import (
    CLASS_MAPPING
)
from segment_utils.image_processing import (
    COLORS,
    GRADIENT_COLORS
    
)

logger = logging.getLogger(__name__)

def main(args):
    save_dir = args.save_dir
    model_name = args.model_name
    attention_mode = args.attention_mode
    video_path = args.video_path
    model_path = args.model_dir / Path(model_name, "best_tumor_iou_model.pth")
    output_video_path = save_dir / Path("segment_video", model_name)
    class_num = len(CLASS_MAPPING)
    # COCOデータセットで事前学習されたFCN-ResNet50モデルをロード
    model = setup_fcn_model(model_name, attention_mode, num_classes=class_num)
    # デバイスの設定（GPUが利用可能なら使用）
    device, model = setup_device(model, model_path=model_path)

    # モデルを推論モードに設定
    model.eval()
    # 動画のセグメンテーションを実行
    process_video(video_path, output_video_path, model, device, transform)

# RGB から BGR への変換関数
def rgb_to_bgr(color):
    return (color[2], color[1], color[0])

def process_video(
    video_path: Path, 
    output_video_path: Path, 
    model: nn.Module, 
    device: torch.device, 
    transform: Callable[[Image.Image], Tensor]
):
    corors_bgr = {class_id: rgb_to_bgr(color) for class_id, color in COLORS.items()}
    cap = cv2.VideoCapture(str(video_path))
    # 動画の基本情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 出力する動画ファイルの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        print(f"Processing frame {frame_idx}/{total_frames}")

        # フレームをPILの画像に変換
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 前処理
        input_tensor = transform(img)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # 推論
        with torch.no_grad():
            output = model(input_batch)['out']

        # 無視するクラス（背景）
        IGNORED_CLASS = 0

        # 各ピクセルに最も確率の高いクラスを割り当てる
        output_predictions = output.argmax(1).squeeze().cpu().numpy()
        unique_values = np.unique(output_predictions)
        logger.info(f"Unique values in output_predictions:{unique_values}")
        # セグメンテーションマスクを作成（4チャンネル：BGRA）
        segmentation_mask = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 4), dtype=np.uint8)
        for class_id, color in corors_bgr.items():
            if class_id != IGNORED_CLASS:
                mask = output_predictions == class_id
                segmentation_mask[mask] = color + (128,)  # 元の色に半透明のアルファチャンネルを追加

        # segmentation_maskのサイズをframeに合わせる
        segmentation_mask = cv2.resize(segmentation_mask, (frame.shape[1], frame.shape[0]))

        print("frame shape:", frame.shape)
        print("segmentation_mask shape:", segmentation_mask.shape)

        # フレームをBGRAに変換
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # マスクを適用
        alpha_channel = segmentation_mask[:, :, 3] / 255.0
        for c in range(3):  # BGRチャンネルに対して
            frame_bgra[:, :, c] = frame_bgra[:, :, c] * (1 - alpha_channel) + segmentation_mask[:, :, c] * alpha_channel

        # 結果を表示（オプション）
        cv2.imshow('Segmentation', frame_bgra)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # BGRに戻して動画ファイルに書き込む（必要な場合）
        blended_frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        out.write(blended_frame_bgr)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def parse_args():
    # オプションの解析
    parser = argparse.ArgumentParser(description="骨格データの生成")

    parser.add_argument("--model_name",
                        type=str,
                        default="fcn_resnet50",
                        choices=["fcn_resnet50", "fcn_resnet101", "fcn_vgg16", "fcn_vgg19", "deeplabv3_resnet101"],
                        help="Choose the model architecture. Available options are: fcn_resnet50, fcn_resnet101, fcn_vgg16, fcn_vgg19, deeplabv3_resnet101."
                        )
    parser.add_argument('--attention_mode', 
                        type=str,
                        default="none",
                        choices=["none", "self_attention", "channel_attention", "both"],
                        help='attention_layerの使用するかを指定する変数'
                        )
    parser.add_argument("--video_path",
                        type=Path,
                        default="../../data/video/test_video.mp4",
                        help='入力データのディレクトリパス'
                        )
    parser.add_argument("--save_dir",
                        type=Path,
                        default="../../result/nagoya",
                        help='結果を保存するディレクトリパス'
                        )
    parser.add_argument("--model_dir",
                        type=Path,
                        default="../../result/nagoya/demo_model",
                        help='転移学習モデルパラメータのパス'
                        )    
    parser.add_argument(
                        '--loglevel',
                        default='INFO',  # デフォルトのログレベルをINFOに設定
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='ログのレベルを設定。'
                        )
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    logger.setLevel(args.loglevel.upper())
    logger.info("loglevel: %s", args.loglevel)
    lformat = "%(name)s <L%(lineno)s> [%(levelname)s] %(message)s"
    logging.basicConfig(
        filename='test_segment.txt',  # 出力先ファイルを指定
        level=logging.INFO,
        filemode='w',  # ファイルを上書きモードに設定
        format=lformat,
    )
    logger.setLevel(args.loglevel.upper())


    main(args)
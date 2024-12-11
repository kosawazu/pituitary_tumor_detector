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
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import Tensor
import time
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
    # 閾値を定義（腫瘍グラデーション）
    thresholds = [0.85, 0.90, 0.95]
    colors_bgr = {class_id: rgb_to_bgr(color) for class_id, color in COLORS.items()}
    gradient_colors_bgr = [rgb_to_bgr(color) for color in GRADIENT_COLORS]
    cap = cv2.VideoCapture(str(video_path))
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 出力動画のサイズを2倍の幅に設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width*2, height))

    frame_time = 1.0 / fps
    last_process_time = time.time()
    frame_idx = 0
    last_segmentation_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - last_process_time

        frame_idx += 1

        if elapsed_time >= 1.0 or last_segmentation_frame is None:
            original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            print(f"Processing frame {frame_idx}/{total_frames}")

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img)
            input_batch = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_batch)['out']

            output_resized = F.interpolate(output, size=(frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=False)
            output_predictions = output_resized.argmax(1).squeeze().cpu().numpy()
            # リサイズした出力を確率に変換
            probabilities = F.softmax(output_resized, dim=1)
            # 確率をCPUに移動し、NumPy配列に変換
            probabilities = probabilities.cpu().numpy()
            #最も確率の高い値の取得
            max_prob = probabilities[0].max(axis=0)

            segmentation_mask = np.zeros((*output_predictions.shape, 3), dtype=np.uint8)
            for class_id, color in colors_bgr.items():
                if class_id == 0:
                    continue
                elif class_id == 4:
                    mask = output_predictions == class_id
                    segmentation_mask[mask & (max_prob < thresholds[0])] = gradient_colors_bgr[0]
                    segmentation_mask[mask & (max_prob >= thresholds[0]) & (max_prob < thresholds[1])] = gradient_colors_bgr[1]
                    segmentation_mask[mask & (max_prob >= thresholds[1])] = gradient_colors_bgr[2]
                else:  # Ignore background class
                    mask = output_predictions == class_id
                    segmentation_mask[mask] = color[::-1]  # BGRをRGBに変換

            # セグメンテーションマスクをPIL形式に変換
            segmentation_image = Image.fromarray(segmentation_mask)

            # セグメンテーション結果のサイズを元画像に合わせる
            segmentation_image_resized = segmentation_image.resize(original_image.size, resample=Image.NEAREST)

            # 元画像とセグメンテーション画像を重ね合わせ
            blended_image = Image.blend(original_image, segmentation_image_resized, alpha=0.4)

            last_segmentation_frame = cv2.cvtColor(np.array(blended_image), cv2.COLOR_RGB2BGR)
            last_process_time = current_time

        # Use the last segmentation result if we're not processing this frame
        if last_segmentation_frame is not None:
            segmentation_frame = last_segmentation_frame
        else:
            segmentation_frame = frame  

        # 元の動画とセグメンテーション結果を横に並べる
        combined_frame = np.hstack((frame, cv2.cvtColor(segmentation_frame, cv2.COLOR_BGRA2BGR)))


        cv2.imshow('Original vs Segmentation', combined_frame)
        out.write(combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
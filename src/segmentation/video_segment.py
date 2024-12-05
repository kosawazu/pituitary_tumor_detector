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

from segment_utils.dataset_utils import(
    transform,
)

from utils.model_utils import (
    setup_fcn_model,
    setup_device,
)

logger = logging.getLogger(__name__)

def main(args):
    save_dir = args.save_dir
    video_path = args.data_dir / Path("video", "video4.avi")
    model_path = args.model_path
    segment_save_dir = save_dir / Path("segment", "video", "fcn_resnet50")
    output_video_path = save_dir / Path("segment_video", "video", "fcn_resnet50")
    os.makedirs(segment_save_dir, exist_ok=True)
    class_labels = ['back', 'damage', 'normal']
    # COCOデータセットで事前学習されたFCN-ResNet50モデルをロード
    model = setup_fcn_model("fcn_resnet50")

    # モデルを推論モードに設定
    model.eval()

    # デバイスの設定（GPUが利用可能なら使用）
    print(f"{model_path=}")
    device, model = setup_device(model, model_path)

    # 動画のセグメンテーションを実行
    process_video(video_path, output_video_path, model, transform, device, class_labels, segment_save_dir)

def process_video(
    video_path: Path, 
    output_video_path: Path, 
    model: nn.Module, 
    device: torch.device, 
    transform: Callable[[Image.Image], Tensor]
):
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

        # 各ピクセルに最も確率の高いクラスを割り当てる
        output_predictions = output.argmax(1).squeeze().cpu().numpy()

        # セグメント結果をカラーマップで表示
        segmented_frame = cv2.applyColorMap((output_predictions * (255 / output_predictions.max())).astype(np.uint8), cv2.COLORMAP_JET)

        # 元の画像とセグメント結果を重ね合わせる
        alpha = 0.5  # 透明度
        blended_frame = cv2.addWeighted(frame, 1 - alpha, segmented_frame, alpha, 0)

        # 結果を表示（オプション）
        cv2.imshow('Segmentation', blended_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 結果を動画ファイルに書き込む
        out.write(blended_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def parse_args():
    # オプションの解析
    parser = argparse.ArgumentParser(description="骨格データの生成")

    parser.add_argument("--video_path",
                        type=Path,
                        default="../../data/video/",
                        help='入力データのディレクトリパス'
                        )
    parser.add_argument("--save_dir",
                        type=Path,
                        default="../../../result/",
                        help='結果を保存するディレクトリパス'
                        )
    parser.add_argument("--model_path",
                        type=Path,
                        default="../../../model/fcn_resnet50/best_model_gear.pth",
                        help='転移学習モデルパラメータのパス'
                        )    
    parser.add_argument("--class_num",
                        type=int,
                        default=3,
                        help='分類するクラス数'
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
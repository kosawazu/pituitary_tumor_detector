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
from screeninfo import get_monitors
from typing import Callable, Tuple, List, Dict

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
    # モデルの設定
    model_name = args.model_name
    attention_mode = args.attention_mode
    model_path = args.model_dir / Path(model_name, "best_tumor_iou_model.pth")
    class_num = len(CLASS_MAPPING)
    model = setup_fcn_model(model_name, attention_mode, num_classes=class_num)
    device, model = setup_device(model, model_path=model_path)
    model.eval()

    # カメラの初期化
    process_camera_feed(model, device, transform)


def process_camera_feed(model, device, transform):
    cap = cv2.VideoCapture(0)  # カメラを開く
    if not cap.isOpened():
        print("カメラを開けません")
        return

    # 変数の初期化
    thresholds = [0.85, 0.90, 0.95]
    colors_bgr = {class_id: rgb_to_bgr(color) for class_id, color in COLORS.items()}
    gradient_colors_bgr = [rgb_to_bgr(color) for color in GRADIENT_COLORS]
    is_paused = False
    last_process_time = time.time()

    cv2.namedWindow('Segmentation Result', cv2.WINDOW_NORMAL)

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            if current_time - last_process_time >= 1.0:  # 1秒ごとに処理
                # セグメンテーション処理
                segmentation_frame = process_frame(
                    frame, model, device, transform, 
                    thresholds, colors_bgr, gradient_colors_bgr
                )
                last_process_time = current_time

                # 結果の表示
                cv2.imshow('Segmentation Result', segmentation_frame)

        # キー入力の処理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # スペースキー
            is_paused = not is_paused

    cap.release()
    cv2.destroyAllWindows()


# RGB から BGR への変換関数
def rgb_to_bgr(color):
    return (color[2], color[1], color[0])

# def initialize_video(
#     video_path: Path
# ) -> Tuple[cv2.VideoCapture, int, int, int, int]:
#     cap = cv2.VideoCapture(str(video_path))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     return cap, width, height, fps, total_frames

# def setup_window(
#     width: int, 
#     height: int
# ) -> None:
#     screen_width, screen_height = get_screen_resolution()
#     cv2.namedWindow('Original vs Segmentation', cv2.WINDOW_NORMAL)
#     initial_width = min(screen_width, width * 2)
#     initial_height = int(height * (initial_width / (width * 2)))
#     cv2.resizeWindow('Original vs Segmentation', initial_width, initial_height)

# def create_video_writer(
#     output_path: Path, 
#     width: int, height: int, 
#     fps: int
# ) -> cv2.VideoWriter:
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     return cv2.VideoWriter(str(output_path), fourcc, fps, (width*2, height))

def process_frame(
    frame: np.ndarray, 
    model: nn.Module, 
    device: torch.device, 
    transform: Callable[[Image.Image], Tensor], 
    thresholds: List[float], 
    colors_bgr: Dict[int, Tuple[int, int, int]], 
    gradient_colors_bgr: List[Tuple[int, int, int]]
) -> np.ndarray:
    original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)['out']

    output_resized = F.interpolate(output, size=(frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=False)
    output_predictions = output_resized.argmax(1).squeeze().cpu().numpy()
    probabilities = F.softmax(output_resized, dim=1).cpu().numpy()
    max_prob = probabilities[0].max(axis=0)

    segmentation_mask = create_segmentation_mask(output_predictions, max_prob, thresholds, colors_bgr, gradient_colors_bgr)
    segmentation_image = Image.fromarray(segmentation_mask)
    segmentation_image_resized = segmentation_image.resize(original_image.size, resample=Image.NEAREST)
    blended_image = Image.blend(original_image, segmentation_image_resized, alpha=0.4)

    return cv2.cvtColor(np.array(blended_image), cv2.COLOR_RGB2BGR)

def create_segmentation_mask(
    output_predictions: np.ndarray, 
    max_prob: np.ndarray, 
    thresholds: List[float], 
    colors_bgr: Dict[int, Tuple[int, int, int]], 
    gradient_colors_bgr: List[Tuple[int, int, int]]
) -> np.ndarray:
    segmentation_mask = np.zeros((*output_predictions.shape, 3), dtype=np.uint8)
    for class_id, color in colors_bgr.items():
        if class_id == 0:
            continue
        elif class_id == 4:
            mask = output_predictions == class_id
            segmentation_mask[mask & (max_prob < thresholds[0])] = gradient_colors_bgr[0]
            segmentation_mask[mask & (max_prob >= thresholds[0]) & (max_prob < thresholds[1])] = gradient_colors_bgr[1]
            segmentation_mask[mask & (max_prob >= thresholds[1])] = gradient_colors_bgr[2]
        else:
            mask = output_predictions == class_id
            segmentation_mask[mask] = color[::-1]
    return segmentation_mask

# def resize_frame(frame: np.ndarray, window_name: str) -> np.ndarray:
#     window_width, window_height = cv2.getWindowImageRect(window_name)[2:4]
#     frame_height, frame_width = frame.shape[:2]
    
#     # アスペクト比を計算
#     aspect_ratio = frame_width / frame_height
#     window_ratio = window_width / window_height

#     if window_ratio > aspect_ratio:
#         # ウィンドウが画像より横長の場合
#         new_height = window_height
#         new_width = int(new_height * aspect_ratio)
#     else:
#         # ウィンドウが画像より縦長の場合
#         new_width = window_width
#         new_height = int(new_width / aspect_ratio)

#     # フレームをリサイズ
#     resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

#     # 黒い背景を作成
#     background = np.zeros((window_height, window_width, 3), dtype=np.uint8)

#     # リサイズしたフレームを中央に配置
#     y_offset = (window_height - new_height) // 2
#     x_offset = (window_width - new_width) // 2
#     background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

#     return background

# def process_video(
#     video_path: Path, 
#     output_video_path: Path, 
#     model: nn.Module, 
#     device: torch.device, 
#     transform: Callable[[Image.Image], Tensor]
# ) -> None:
#     thresholds = [0.85, 0.90, 0.95]
#     colors_bgr = {class_id: rgb_to_bgr(color) for class_id, color in COLORS.items()}
#     gradient_colors_bgr = [rgb_to_bgr(color) for color in GRADIENT_COLORS]

#     cap, width, height, fps, total_frames = initialize_video(video_path)
#     setup_window(width, height)  # ウィンドウサイズは処理済み映像に合わせる
#     out = create_video_writer(output_video_path, width, height, fps)

#     frame_time = 1.0 / fps
#     last_process_time = time.time()
#     frame_idx = 0
#     last_segmentation_frame = None
#     is_paused = False

#     while cap.isOpened():
#         if not is_paused:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             current_time = time.time()
#             elapsed_time = current_time - last_process_time
#             frame_idx += 1

#             if elapsed_time >= 1.0 or last_segmentation_frame is None:
#                 print(f"Processing frame {frame_idx}/{total_frames}")
#                 last_segmentation_frame = process_frame(frame, model, device, transform, thresholds, colors_bgr, gradient_colors_bgr)
#                 last_process_time = current_time

#             segmentation_frame = last_segmentation_frame if last_segmentation_frame is not None else frame
#             resized_frame = resize_frame(segmentation_frame, 'Original vs Segmentation')  # 処理済み映像のみリサイズ

#             cv2.imshow('Original vs Segmentation', resized_frame)  # 処理済み映像のみ表示
#             out.write(segmentation_frame)  # 処理済み映像のみ保存

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == 32:  # スペースキーのASCIIコード
#             is_paused = not is_paused
#             print("Paused" if is_paused else "Resumed")
#         elif key == ord('n') and is_paused:
#             # 一時停止中に'n'キーを押すと次のフレームに進む
#             ret, frame = cap.read()
#             if ret:
#                 frame_idx += 1
#                 print(f"Advancing to frame {frame_idx}/{total_frames}")
#                 last_segmentation_frame = process_frame(frame, model, device, transform, thresholds, colors_bgr, gradient_colors_bgr)
#                 segmentation_frame = last_segmentation_frame
#                 resized_frame = resize_frame(segmentation_frame, 'Original vs Segmentation')
#                 cv2.imshow('Original vs Segmentation', resized_frame)
#                 out.write(segmentation_frame)

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


def get_screen_resolution():
    """画面の解像度を取得する関数"""
    try:
        monitor = get_monitors()[0]
        return monitor.width, monitor.height
    except:
        # スクリーン情報を取得できない場合はデフォルト値を返す
        return 1920, 1080

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
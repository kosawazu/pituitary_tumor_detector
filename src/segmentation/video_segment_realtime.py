import os
import argparse
import logging
from pathlib import Path
import sys
import cv2
import torch
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 自作モジュール
from segmentation import transform
from utils.model_utils import setup_device, tune_model
from torchvision import models
from segment_utils.dataset_utils import CLASS_MAPPING
from segment_utils.image_processing import COLORS, GRADIENT_COLORS
logger = logging.getLogger(__name__)
def main(args):
    # モデルの設定
    model_name = args.model_name
    attention_mode = args.attention_mode
    # PyInstaller 実行時と通常実行時の base_path の設定
    if getattr(sys, 'frozen', False):  # PyInstaller 実行時
        base_path = sys._MEIPASS
        logger.info("Running in PyInstaller environment.")
    else:  # 通常のスクリプト実行時
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # src の親ディレクトリを基準に設定
        logger.info("Running in standard Python environment.")
    # モデルファイルのパスを取得
    model_path = os.path.join(base_path, "result/nagoya/demo_model/deeplabv3_resnet101/best_tumor_iou_model.pth")
    # モデルファイルの存在確認
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    else:
        logger.info(f"Model file found at: {model_path}")
    class_num = len(CLASS_MAPPING)
    model = models.segmentation.deeplabv3_resnet101(pretrained=False)
    model = tune_model(model, model_name, attention_mode, num_classes=class_num)
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
                        default="deeplabv3_resnet101",
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
    # モデルのロード
    logger.info(f"Arguments: {args}")
    main(args)

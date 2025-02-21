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
from utils.model_utils_conf import setup_device, tune_model
from torchvision import models
from segment_utils.dataset_utils import CLASS_MAPPING
from segment_utils.image_processing import COLORS, GRADIENT_COLORS
import tkinter as tk
from tkinter import messagebox
from pygrabber.dshow_graph import FilterGraph


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
        # base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # src の親ディレクトリを基準に設定
        base_path = "../../"  # src の親ディレクトリを基準に設定
        logger.info("Running in standard Python environment.")
    model_path = "../../result/nagoya/demo_model/deeplabv3_resnet101/best_tumor_iou_model.pth"
    print(f"{model_path=}")
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
    video_path = "../../data/video/test_video.mp4"
    model.eval()
    # カメラの初期化
    process_camera_feed(model, device, transform, video_path)

def get_camera_list():
    """WindowsのDirectShowを利用してカメラの名称を取得"""
    graph = FilterGraph()
    return graph.get_input_devices()

def get_camera_name(camera_id):
    """カメラの名称を取得する"""
    cameras = get_camera_list()
    if 0 <= camera_id < len(cameras):
        return cameras[camera_id]
    return f"Unknown Camera {camera_id}"

def list_cameras_with_names(max_test=5):
    """利用可能なカメラの ID と名称を取得"""
    available_cameras = []
    cameras = get_camera_list()
    
    for i in range(min(len(cameras), max_test)):  # カメラの数に制限
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow を利用
        if cap.isOpened():
            available_cameras.append((i, cameras[i]))  # (ID, 名前)
            cap.release()
            time.sleep(0.1)
    
    return available_cameras

def select_camera_gui():
    """GUIでカメラを選択するウィンドウを表示"""
    cameras = list_cameras_with_names()
    if not cameras:
        messagebox.showerror("エラー", "利用可能なカメラが見つかりません")
        return None

    def on_select():
        selected_idx = listbox.curselection()
        if not selected_idx:
            messagebox.showwarning("警告", "カメラを選択してください")
            return
        selected_camera["id"] = cameras[selected_idx[0]][0]  # ID を保存
        root.destroy()

    root = tk.Tk()
    root.title("カメラ選択")

    # 画面サイズの取得
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # ウィンドウサイズを画面の半分に設定
    window_width = screen_width // 2
    window_height = screen_height // 2
    root.geometry(f"{window_width}x{window_height}+{window_width//2}+{window_height//2}")  # 画面中央に配置

    tk.Label(root, text="使用するカメラを選択").pack(pady=10)
    listbox = tk.Listbox(root, selectmode=tk.SINGLE)
    for _, name in cameras:
        listbox.insert(tk.END, name)  # カメラ名を表示
    listbox.pack(pady=10, fill=tk.BOTH, expand=True)

    tk.Button(root, text="選択", command=on_select).pack(pady=10)
    
    selected_camera = {"id": None}
    root.mainloop()
    return selected_camera["id"]

def process_camera_feed(model, device, transform, video_path):
    """GUIを使ってカメラを選択し、そのカメラで処理を行う"""
    while True:
        # camera_id = select_camera_gui()
        # if camera_id is None:
        #     return  # カメラがない場合は終了
        
        cap = cv2.VideoCapture(video_path)
        
        is_paused = False  # 一時停止フラグ
        last_segmentation_frame = None
        last_process_time = time.time()
        cv2.namedWindow('Segmentation Result', cv2.WINDOW_NORMAL)

        while cap.isOpened():
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("フレームを取得できません")
                    break
                current_time = time.time()
                elapsed_time = current_time - last_process_time

                if elapsed_time >= 1.0 or last_segmentation_frame is None:
                    last_segmentation_frame = process_frame(frame, model, device, transform)
                    last_process_time = current_time
                
                # セグメンテーション処理を行い、結果を得る
                segmentation_frame = last_segmentation_frame if last_segmentation_frame is not None else frame
                cv2.imshow('Segmentation Result', segmentation_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # `q` で終了
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == 32:  # スペースキーで一時停止 / 再開
                is_paused = not is_paused

        cap.release()
        cv2.destroyAllWindows()

# RGB から BGR への変換関数
def rgb_to_bgr(color):
    return (color[2], color[1], color[0])

def process_frame(frame, model, device, transform):
    """フレームをセグメンテーション処理"""
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCVからRGB変換
    img = Image.fromarray(original_image)
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out']

    # 出力結果をサイズ変更
    output_resized = F.interpolate(output, size=(frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=False)
    output_predictions = output_resized.argmax(1).squeeze().cpu().numpy()
    max_prob = F.softmax(output_resized, dim=1).max(1)[0].squeeze().cpu().numpy()  # 最大確率を取得

    # 閾値、色、グラデーションを設定
    thresholds = [0.3, 0.6]  # 例: 閾値の設定
    colors_bgr = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}  # クラスごとの色
    gradient_colors_bgr = [(230, 230, 250), (128, 0, 128), (75, 0, 130)]  # グラデーション色

    # 結果のマスク作成とカラー処理
    segmentation_mask = create_segmentation_mask(output_predictions, max_prob, thresholds, colors_bgr, gradient_colors_bgr)
    segmentation_image = Image.fromarray(segmentation_mask)

    # 元の画像に重ね合わせ
    blended_image = Image.blend(img, segmentation_image, alpha=0.4)
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

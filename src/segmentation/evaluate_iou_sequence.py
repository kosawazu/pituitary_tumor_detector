import os
import argparse
import logging
from pathlib import Path
import json
import torch
import sys
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import re
from typing import Tuple, List
sys.path.append("../")
# 自作モジュール

from segment_utils.dataset_utils import(
    get_transform,
    CLASS_MAPPING,
    TARGET_CLASS_MAPPING,
)

from segment_utils.image_processing import(
    segment_save,
    save_blended_image,
    save_blended_image_with_class4_gradient,
    resize_segmentation_tensor
)

from segment_utils.graph import(
    plot_iou_by_image
)

from segment_utils.metrics import(
    calculate_target_class_iou
)

from utils.model_utils import (
    setup_model,
    setup_device
)

from utils.general_utils import (
    tuple_type
)

logger = logging.getLogger(__name__)

def main(args):
    image_dir = args.image_dir 
    class_num = args.class_num
    model_name = args.model_name
    save_dir = args.save_dir
    model_path = args.model_path
    segment_save_dir = save_dir / Path("segment_image")
    blended_segment_save_dir = save_dir / Path("blend_image")
    metrics_segment_save_dir = save_dir / Path("metrics")
    gradation_segment_save_dir = save_dir / Path("blend_gradation_image")
    image_paths = list(image_dir.glob("*.jpg"))
    test_image_paths, test_true_labels = get_sorted_test_labels(args.json_path, image_paths)
    transform = get_transform(args.image_size)
    os.makedirs(segment_save_dir, exist_ok=True)
    os.makedirs(blended_segment_save_dir, exist_ok=True)
    os.makedirs(metrics_segment_save_dir, exist_ok=True)
    os.makedirs(gradation_segment_save_dir, exist_ok=True)
    class_names = [CLASS_MAPPING[i] for i in sorted(CLASS_MAPPING.keys())]
    target_class_names = [TARGET_CLASS_MAPPING[i] for i in sorted(TARGET_CLASS_MAPPING.keys())]
    """モデルをデバイス（GPU/CPU）に設定し、必要に応じてマルチGPUモードに切り替えます。"""
    # COCOデータセットで事前学習されたFCN-ResNet50モデルをロード
    logger.info(f"{model_path}を読み込みます")
    model = setup_model(model_name, num_classes=class_num)
    # デバイスの設定（GPUが利用可能なら使用）
    device, model = setup_device(model, model_path=model_path)

    # モデルを推論モードに設定
    model.eval()

    # 画像ごとのIoUを保存するための辞書を作成
    image_iou_dict = {}
    for test_image_path, test_image_label in zip(test_image_paths, test_true_labels):
        image_name = test_image_path.stem
        test_image_label_tensor = torch.tensor(test_image_label, device=device)
        # 入力画像を読み込み、前処理
        img = Image.open(test_image_path).convert('RGB')
        input_tensor = transform(img)
        input_batch = input_tensor.unsqueeze(0).to(device)  # バッチ次元を追加

        # 推論
        with torch.no_grad():
            output = model(input_batch)
            if isinstance(output, dict):
                output = output['out']

        # 各ピクセルに最も確率の高いクラスを割り当てる
        output_resized = resize_segmentation_tensor(output, test_image_label.shape[-2:])
        output_predictions = output_resized.argmax(1).squeeze().cpu().numpy()
        ious = calculate_target_class_iou(output_predictions, test_image_label_tensor)

        image_iou_dict[image_name] = ious
        # #セグメントした画像を保存
        segment_save(segment_save_dir, test_image_path, output_predictions)
        save_blended_image(blended_segment_save_dir, test_image_path, output_predictions)
        save_blended_image_with_class4_gradient(gradation_segment_save_dir, test_image_path, output)

    # グラフ描画
    plot_iou_by_image(image_iou_dict, target_class_names, save_dir)
    
def natural_sort_key(s):
    """自然順ソート用のキー関数"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def get_sorted_test_labels(
    json_path: str,
    image_paths: List[Path]
) -> Tuple[List[Path], List[np.ndarray]]:
    """
    VoTTラベルJSONファイルから指定された画像パスに基づいてラベルを生成し、
    ファイル名で自然順ソートした画像パスとラベルのペアを返す関数
    """
    # 元の関数でラベルを取得
    unsorted_image_paths, unsorted_labels = get_test_labels(json_path, image_paths)
    
    # パスとラベルをペアにする
    path_label_pairs = list(zip(unsorted_image_paths, unsorted_labels))
    
    # 自然順でソート
    sorted_pairs = sorted(path_label_pairs, key=lambda x: natural_sort_key(x[0].name))
    
    # ソートされたパスとラベルを別々のリストに戻す
    sorted_image_paths = [pair[0] for pair in sorted_pairs]
    sorted_labels = [pair[1] for pair in sorted_pairs]
    
    return sorted_image_paths, sorted_labels

def get_test_labels(
    json_path: str,
    image_paths: List[Path]
) -> Tuple[List[Path], List[np.ndarray]]:
    """
    単一のVoTTラベルJSONファイルから、指定された画像パスリストに基づいてラベルを生成する関数
    """
    result_image_paths = []
    labels = []
    
    # 画像名のリストを準備
    image_names = [p.name for p in image_paths]
    
    # 画像パスを名前で検索するための辞書を作成
    image_path_by_name = {p.name: p for p in image_paths}
    
    # JSONファイルが存在するか確認
    if not Path(json_path).exists():
        print(f"Warning: JSON file {json_path} not found")
        return result_image_paths, labels
    
    # JSONファイルを読み込む
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 画像名をキーとしてassetsを検索しやすくする辞書を作成
    assets_by_name = {}
    for asset_id, asset_data in data.get('assets', {}).items():
        if 'asset' in asset_data and 'name' in asset_data['asset']:
            assets_by_name[asset_data['asset']['name']] = asset_data
    
    # 各画像名に対して処理
    for img_filename in image_names:
        # この画像名に該当するアセットを取得
        if img_filename not in assets_by_name:
            print(f"Image {img_filename} not found in JSON data, skipping.")
            continue
        
        # 対応する画像パスを取得
        img_path = image_path_by_name[img_filename]
        
        # アセットデータを取得
        asset_data = assets_by_name[img_filename]
        img_metadata = asset_data['asset']
        
        # アノテーション領域を読み込む
        width = img_metadata['size']['width']
        height = img_metadata['size']['height']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # アノテーション情報からマスクを作成
        regions = asset_data.get('regions', [])
        for region in regions:
            points = region.get('points', [])
            if not points:
                continue
                
            polygon = [(point['x'], point['y']) for point in points]
            
            # ポリゴンをバイナリマスクに変換
            img_mask = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img_mask).polygon(polygon, outline=1, fill=1)
            region_mask = np.array(img_mask)
            
            # カテゴリごとにマスクを作成（カテゴリ名で対応付け）
            tags = region.get('tags', [])
            if "sellar" in tags:
                mask = np.maximum(mask, region_mask * 1)  # クラスID 1を使用
            elif "sella" in tags:
                mask = np.maximum(mask, region_mask * 2)  # クラスID 2を使用
            elif "pituitary" in tags:
                mask = np.maximum(mask, region_mask * 3)  # クラスID 3を使用
            elif "tumor" in tags:
                mask = np.maximum(mask, region_mask * 4)  # クラスID 4を使用
        
        # 画像パスとラベルをそれぞれのリストに追加
        result_image_paths.append(img_path)
        labels.append(mask)
    
    return result_image_paths, labels

def parse_args():
    # オプションの解析
    parser = argparse.ArgumentParser(description="セグメンテーションモデルの推論")

    parser.add_argument("--image_dir",
                        type=Path,
                        help='入力データのディレクトリパス'
                        )
    parser.add_argument("--json_path",
                        type=Path,
                        help='入力データのディレクトリパス'
                        )
    parser.add_argument("--save_dir",
                        type=Path,
                        default="../../result/ext_val",
                        help='結果を保存するディレクトリパス'
                        )  
    parser.add_argument("--model_path",
                        type=Path,
                        help='結果を保存するディレクトリパス'
                        ) 
    parser.add_argument("--model_name",
                        type=str,
                        default="deeplabv3_resnet101",
                        choices=["fcn_resnet50", "fcn_resnet101", "deeplabv3_resnet101", "vit_b_16_segmentation", "fcn_bot_resnet101"],
                        help="Choose the model architecture. Available options are: fcn_resnet50, fcn_resnet101, fcn_bot_resnet101, deeplabv3_resnet101, vit_b_16_segmentation."
                        )
    parser.add_argument("--class_num",
                        type=int,
                        default=5,
                        help='分類するクラス数'
                        )
    parser.add_argument("--image_size",
                        type=tuple_type,
                        default=(224, 224),
                        help='画像サイズ (height, width)'
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
        level=logging.INFO,
        filemode='w',  # ファイルを上書きモードに設定
        format=lformat,
    )
    logger.setLevel(args.loglevel.upper())


    main(args)
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
from typing import Tuple, List
sys.path.append("../")
# 自作モジュール

from segment_utils.dataset_utils import(
    get_transform,
    CLASS_MAPPING,
)

from segment_utils.image_processing import(
    segment_save,
    save_blended_image,
    save_blended_image_with_class4_gradient,
    resize_segmentation_tensor
)

from segment_utils.metrics import(
    save_iou_to_csv,
    save_confusion_matrix_with_metrics,
    calculate_iou,
    update_ious_and_counts,
    calculate_average_ious_and_miou,
    calculate_iou_and_miou_from_confusion_matrix
)

from utils.model_utils import (
    setup_fcn_model,
    setup_device
)

from utils.general_utils import (
    tuple_type
)

logger = logging.getLogger(__name__)

def main(args):
    data_dir = args.data_dir 
    class_num = args.class_num
    model_name = args.model_name
    model_file = args.save_model_file
    model_data_dir = args.save_dir / Path("training_results", model_name, str(args.batch_size), "{:.1e}".format(args.learning_rate))
    test_text_file_path = model_data_dir / Path(f"others/split_dataset/val_filenames.txt")
    test_result_dir = model_data_dir / Path("test", model_file)
    segment_save_dir = test_result_dir / Path("segment_image")
    blended_segment_save_dir = test_result_dir / Path("blend_image")
    metrics_segment_save_dir = test_result_dir / Path("metrics")
    gradation_segment_save_dir = test_result_dir / Path("blend_gradation_image")
    model_path = model_data_dir / Path("model", f"{model_file}.pth")
    file_names_list = get_test_image_name(test_text_file_path)
    test_image_paths, test_true_labels = get_test_image_paths_and_labels(file_names_list, data_dir)
    transform = get_transform(args.image_size)
    os.makedirs(segment_save_dir, exist_ok=True)
    os.makedirs(blended_segment_save_dir, exist_ok=True)
    os.makedirs(metrics_segment_save_dir, exist_ok=True)
    os.makedirs(gradation_segment_save_dir, exist_ok=True)
    class_names = [CLASS_MAPPING[i] for i in sorted(CLASS_MAPPING.keys())]
    num_classes = len(class_names)
    """モデルをデバイス（GPU/CPU）に設定し、必要に応じてマルチGPUモードに切り替えます。"""
    # COCOデータセットで事前学習されたFCN-ResNet50モデルをロード
    logger.info(f"{model_path}を読み込みます")
    model = setup_fcn_model(model_name, num_classes=class_num)
    # デバイスの設定（GPUが利用可能なら使用）
    device, model = setup_device(model, model_path=model_path)

    # モデルを推論モードに設定
    model.eval()

    all_ground_truths = []
    all_predictions = []
    all_ious = []
    all_iou_counts = []
    for test_image_path, test_image_label in zip(test_image_paths, test_true_labels):
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
        ious = calculate_iou(output_predictions, test_image_label_tensor, num_classes)
        all_ious, all_iou_counts = update_ious_and_counts(all_ious, all_iou_counts, ious)

        # ピクセルごとに予測ラベルと正解ラベルをフラットにする
        all_ground_truths.append(test_image_label.flatten())
        all_predictions.append(output_predictions.flatten())
        #セグメントした画像を保存
        segment_save(segment_save_dir, test_image_path, output_predictions)
        save_blended_image(blended_segment_save_dir, test_image_path, output_predictions)
        save_blended_image_with_class4_gradient(gradation_segment_save_dir, test_image_path, output)


    # すべての画像の正解ラベルと予測ラベルをまとめる
    all_ground_truths = np.concatenate(all_ground_truths)
    all_predictions = np.concatenate(all_predictions)
    avg_ious, miou = calculate_average_ious_and_miou(all_ious)
    # ピクセル単位の混同行列を作成
    conf_matrix = confusion_matrix(all_ground_truths, all_predictions, labels=list(range(num_classes)))
    # iouを混同行列から計算
    iou_from_matrix, miou_from_matrix = calculate_iou_and_miou_from_confusion_matrix(conf_matrix, num_classes)
    # 混同行列とメトリクスを保存
    save_confusion_matrix_with_metrics(conf_matrix, metrics_segment_save_dir, class_names)
    # CSVにIoU結果を保存
    save_iou_to_csv(avg_ious, miou, class_names, metrics_segment_save_dir, Path("iou_results.csv"), num_classes)
    # 混同行列から計算したiouを保存
    save_iou_to_csv(iou_from_matrix, miou_from_matrix, class_names, metrics_segment_save_dir, Path("iou_results_from_conf_matrix.csv"), num_classes)
    


def get_test_image_name(
    txt_file_path: Path
) -> List[str]:
    # ファイル名を格納するリスト
    file_names_list = []

    # テキストファイルを読み込んでリストに格納
    with open(txt_file_path, 'r') as file:
        # 各行を読み込み、改行を削除してリストに追加
        file_names_list = [line.strip() for line in file]

    return file_names_list

def get_test_image_paths_and_labels(
    file_names_list: List[str],
    data_dir: Path
) -> Tuple[List[Path], List[np.ndarray]]:
    image_paths = []
    labels = []
    image_dir = data_dir / Path("img")
    json_dir = data_dir / Path("mask_json")

    for json_file_name in file_names_list:
        # JSONファイルのフルパスを作成
        json_file_path = json_dir / Path(json_file_name)

        # JSONファイルが存在するか確認
        if not json_file_path.exists():
            print(f"Warning: JSON file {json_file_name} not found in {json_dir}")
            continue  # 次のファイルに進む

        # JSONファイルを読み込む
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # 画像ファイル名を取得
        img_metadata = data['asset']
        img_filename = img_metadata['name']  # ファイル名のみ取得

        # 画像ファイルのフルパスを作成 (image_dir からファイル名を探す)
        img_path = image_dir / img_filename

        # 画像ファイルが存在するか確認
        if not img_path.exists():
            print(f"Image not found: {img_path}, skipping.")
            continue

        # アノテーション領域を読み込む
        mask = np.zeros((img_metadata['size']['height'], img_metadata['size']['width']), dtype=np.uint8)

        # アノテーション情報からマスクを作成
        regions = data.get('regions', [])
        for region in regions:
            points = region['points']
            polygon = [(point['x'], point['y']) for point in points]

            # ポリゴンをバイナリマスクに変換
            img_mask = Image.new('L', (img_metadata['size']['width'], img_metadata['size']['height']), 0)
            ImageDraw.Draw(img_mask).polygon(polygon, outline=1, fill=1)
            region_mask = np.array(img_mask)

            # カテゴリごとにマスクを作成（カテゴリ名で対応付け）
            if "sellar" in region['tags']:
                mask = np.maximum(mask, region_mask * 1)  # クラスID 1を使用
            elif "sella" in region['tags']:
                mask = np.maximum(mask, region_mask * 2)  # クラスID 2を使用
            elif "pituitary" in region['tags']:
                mask = np.maximum(mask, region_mask * 3)  # クラスID 3を使用
            elif "tumor" in region['tags']:
                mask = np.maximum(mask, region_mask * 4)  # クラスID 4を使用

        # 画像パスとラベルをそれぞれのリストに追加
        image_paths.append(img_path)
        labels.append(mask)

    return image_paths, labels

def parse_args():
    # オプションの解析
    parser = argparse.ArgumentParser(description="骨格データの生成")

    parser.add_argument("--data_dir",
                        type=Path,
                        default="../../data/",
                        help='入力データのディレクトリパス'
                        )
    parser.add_argument("--save_dir",
                        type=Path,
                        default="../../result/",
                        help='結果を保存するディレクトリパス'
                        )  
    parser.add_argument("--model_name",
                        type=str,
                        default="fcn_resnet50",
                        choices=["fcn_resnet50", "fcn_resnet101", "deeplabv3_resnet101", "vit_b_16_segmentation", "fcn_bot_resnet101"],
                        help="Choose the model architecture. Available options are: fcn_resnet50, fcn_resnet101, fcn_bot_resnet101, deeplabv3_resnet101, vit_b_16_segmentation."
                        )
    parser.add_argument("--save_model_file",
                        type=str,
                        help='保存したモデルのファイル名'
                        ) 
    parser.add_argument("--batch_size",
                        type=int,
                        default=20
                        )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4
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
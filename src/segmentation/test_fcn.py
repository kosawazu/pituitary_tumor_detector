import os
import argparse
import logging
from pathlib import Path

import torch

from PIL import Image
import matplotlib.pyplot as plt

from typing import Tuple, List

# 自作モジュール
from segmentation import (
    transform, 
    SegmentationDataset,
    segment_save,
    visualize_random_sample_from_dataset
)

from utils.model_utils import (
    setup_fcn_model,
    setup_device
)

logger = logging.getLogger(__name__)

def main(args):
    data_dir = args.data_dir 
    class_num = args.class_num
    save_dir = args.save_dir
    model_dir = args.model_dir
    model_name = args.model_name
    txt_file_path = save_dir / Path(f"training_results/{model_name}/others/split_dataset/test_filenames.txt")
    segment_save_dir = save_dir / Path("segment_image", model_name)
    os.makedirs(segment_save_dir, exist_ok=True)
    model_path = model_dir / Path(model_name , "model", "best_model_segment.pth")
    file_names_list = get_test_image_name(txt_file_path)
    test_image_paths = get_test_image_path(file_names_list, data_dir)
    """モデルをデバイス（GPU/CPU）に設定し、必要に応じてマルチGPUモードに切り替えます。"""
    # COCOデータセットで事前学習されたFCN-ResNet50モデルをロード
    logger.info(model_path)
    model = setup_fcn_model(model_name, num_classes=class_num)
    # デバイスの設定（GPUが利用可能なら使用）
    device, model = setup_device(model, model_path=model_path,)

    # モデルを推論モードに設定
    model.eval()

    for test_image_path in test_image_paths:
        # 入力画像を読み込み、前処理
        img = Image.open(test_image_path).convert('RGB')
        input_tensor = transform(img)
        input_batch = input_tensor.unsqueeze(0).to(device)  # バッチ次元を追加

        # 推論
        with torch.no_grad():
            output = model(input_batch)['out']  # FCNの出力

        # 各ピクセルに最も確率の高いクラスを割り当てる
        output_predictions = output.argmax(1).squeeze().cpu().numpy()

        if 2 not in output_predictions:  # クラスインデックス2（傷）がない場合
        # 「紙袋」の色を緑（クラスインデックス1のまま）に強制
        # ここでは「紙袋」(1)をそのままにして、セグメント保存関数で色をマッピング
        # もしくは配色を変更したい場合はこちらで値を変換できる
            pass

        segment_save(segment_save_dir, test_image_path, output_predictions)

def get_test_image_name(
    txt_file_path
):
    # ファイル名を格納するリスト
    file_names_list = []

    # テキストファイルを読み込んでリストに格納
    with open(txt_file_path, 'r') as file:
        # 各行を読み込み、改行を削除してリストに追加
        file_names_list = [line.strip() for line in file]

    return file_names_list

def get_test_image_path(
    file_names_list: List[str],
    data_dir: Path
) -> List[Path]:
    test_image_paths = []
    # normal と abnormal フォルダのパスを定義
    normal_dir = data_dir / Path("normal")
    abnormal_dir = data_dir / Path("abnormal")
    
    for file_name in file_names_list:
        # normal フォルダでファイルを探索
        normal_file_path = normal_dir / Path(file_name)
        if normal_file_path.exists():
            test_image_paths.append(normal_file_path)
            continue  # 見つかったので次のファイル名へ
        
        # abnormal フォルダでファイルを探索
        abnormal_file_path = abnormal_dir / Path(file_name)
        if abnormal_file_path.exists():
            test_image_paths.append(abnormal_file_path)
            continue  # 見つかったので次のファイル名へ
        
        # ファイルがどちらのフォルダにも見つからなかった場合（オプション）
        logger.info(f"Warning: {file_name} not found in {normal_dir} or {abnormal_dir}")
    
    return test_image_paths

def parse_args():
    # オプションの解析
    parser = argparse.ArgumentParser(description="骨格データの生成")

    parser.add_argument("--data_dir",
                        type=Path,
                        default="../../../data/paper_bag/",
                        help='入力データのディレクトリパス'
                        )
    parser.add_argument("--save_dir",
                        type=Path,
                        default="../../../result/",
                        help='結果を保存するディレクトリパス'
                        )
    parser.add_argument("--model_dir",
                        type=Path,
                        default="../../../result/training_results/",
                        help='転移学習モデルパラメータのパス'
                        )    
    parser.add_argument("--model_name",
                        type=str,
                        default="fcn_resnet50"
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
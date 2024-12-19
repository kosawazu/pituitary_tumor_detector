import os
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import argparse
import logging
import sys
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Tuple, List, Dict

sys.path.append("../")
# 自作モジュール

#データセット関連
from segment_utils.dataset_utils import(
    get_transform,
    SegmentationDataset,
    split_dataset,
    CLASS_MAPPING
)
# 画像処理関連
from segment_utils.image_processing import(
    segment_save,
    resize_mask
)
# 評価関連
from segment_utils.metrics import(
    calculate_iou,
    update_ious_and_counts,
    calculate_average_ious_and_miou,
    calculate_iou_and_miou_from_confusion_matrix
)
# グラフ
from segment_utils.graph import(
    plot_and_save_metrics_curve,
    plot_and_save_iou_curve
)
# 確認用
from segment_utils.misc import(
    visualize_random_sample_from_dataset,
    save_model_architecture
)
#モデル関連
from utils.model_utils import (
    setup_fcn_model,
    setup_device
)
#学習関連
from utils.training_utils import (
    EarlyStopping
)
#seed値固定
from utils.set_seed import (
    seed_everything
)
from utils.general_utils import (
    tuple_type
)

logger = logging.getLogger(__name__)

def main(args):
    # 必要な変数の定義とディレクトリの作成
    data_dir = args.data_dir
    mask_dir = args.mask_dir
    model_name = args.model_name
    save_dir = args.save_dir / Path("nagoya", "training_results", model_name, str(args.batch_size), "{:.1e}".format(args.learning_rate))
    epochs = args.epochs
    class_num = args.class_num
    attention_mode = args.attention_mode
    train_image_dir = data_dir / Path("img")
    model_save_dir = save_dir / Path( "model")
    graph_save_dir = save_dir / Path("graph")
    segment_dir = save_dir / Path("val_segment")
    others_save_dir = save_dir / Path("others")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(graph_save_dir, exist_ok=True)
    os.makedirs(segment_dir, exist_ok=True)
    class_names = ['background', 'sellar', 'sella', 'pituitary', 'tumor']
    """モデルをデバイス（GPU/CPU）に設定し、必要に応じてマルチGPUモードに切り替えます。"""

    # COCOデータセットで事前学習されたモデルをロード
    model = setup_fcn_model(model_name, attention_mode, num_classes=class_num)
    # デバイスの設定（GPUが利用可能なら使用）
    device, model = setup_device(model)
    transform = get_transform(args.image_size)
    # データセットの作成
    dataset = SegmentationDataset(mask_dir, train_image_dir, transform)
    #データ数の確認
    logger.info(f"Total samples in dataset: {len(dataset)}")

    # サンプルを取得して確認
    sample_idx = 0  # 確認したいインデックス
    image, mask, filename = dataset[sample_idx]  # 0番目のサンプルを取得

    if image is None or mask is None:
        logger.debug(f"Sample {sample_idx} has no valid data.")
    # データセットを指定した比率でtrain,val,testに分割をし、それぞれのデータローダーの作成
    train_dataset, valid_dataset, _ = split_dataset(dataset, output_dir=others_save_dir / "split_dataset")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=True, pin_memory=True, num_workers=0)

    # マスクのラベルが指定したクラス数に収まっているか確認する例
    mask = train_dataset[10][1]  # 0番目のサンプルのマスクを取得
    # mask = map_mask_to_four_classes(mask)  # クラスのマッピングを実行
    logger.debug(f'Categories after mapping: {mask.unique()}')  # マスク内のユニークなクラスラベルを確認


    # 実行時のモデルアーキテクチャとデータの可視化(確認用)
    save_model_architecture(model, others_save_dir)
    visualize_random_sample_from_dataset(train_dataset, others_save_dir)

    # 損失関数と最適化手法
    criterion = nn.CrossEntropyLoss()  # セマンティックセグメンテーションの損失関数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 早期終了の定義（patienceで監視するエポック数を指定）
    early_stopping = EarlyStopping(patience=args.patience, verbose=True) 

    #モデルの学習と学習曲線の出力
    train_model(train_loader, valid_loader, device, optimizer, model, criterion, epochs, early_stopping, class_num, class_names, train_image_dir, graph_save_dir, model_save_dir, segment_dir)

    return

def save_epoch_info(model_dir: str, metric_name: str, epoch: int):
    """エポック情報をテキストファイルに保存する"""
    file_path = os.path.join(model_dir, f'best_{metric_name}_epoch.txt')
    with open(file_path, 'w') as f:
        f.write(f"Best {metric_name} model saved at epoch: {epoch + 1}")
    logger.info(f"エポック情報を保存しました: {file_path}")


def save_best_model(
    epoch: int,
    model: nn.Module, 
    metric: Dict[str, float], 
    best_metric: float, 
    model_dir: Path, 
    file_name: Path, 
    is_higher_better: bool,
    metric_name
) -> float:
    """
    最良のモデルを保存する
    """
    should_save = False

    if is_higher_better:
        if metric >= best_metric:
            should_save = True
    else:
        if metric <= best_metric:
            should_save = True

    if should_save:
        best_metric = metric
        file_path = os.path.join(model_dir, file_name)
        torch.save(model.state_dict(), file_path)
        metric_name = file_name.split('_')[1]  # 'best_loss_model.pth' から 'loss' を抽出
        save_epoch_info(model_dir, metric_name, epoch)
        logger.info(f"{metric_name}の最良モデルを保存しました: {file_path=} （{metric=}, {best_metric=}）")
    else:
        logger.info(f"{metric_name}のモデルは保存されませんでした。現行の最良値を維持します。 （{metric=}, {best_metric=})")

    return best_metric

def save_best_models(
    epoch: int,  
    model: nn.Module, 
    metrics: Dict[str, float], 
    best_metrics: Dict[str, float], 
    model_dir: str
) -> Dict[str, float]:
    for metric_name, is_higher_better in [("loss", False), ("tumor_iou", True), ("mean_iou", True)]:
        best_metrics[metric_name] = save_best_model(
            epoch,
            model, 
            metrics[metric_name], 
            best_metrics[metric_name], 
            model_dir, 
            f'best_{metric_name}_model.pth', 
            is_higher_better,
            metric_name
        )
    return best_metrics

def train_model(
    train_loader: DataLoader,   
    valid_loader: DataLoader,   
    device: torch.device,       
    optimizer: optim.Optimizer, 
    model: nn.Module,           
    criterion: nn.Module,       
    epochs: int,
    early_stopping: EarlyStopping,
    class_num: int,
    class_names: List[str],
    data_dir: Path,
    graph_save_dir: Path,
    model_save_dir: Path,
    segment_dir: Path               
) -> None:                  
    """"""  
    logger.info("学習を開始します。")  
    # 損失,iousをエポックごとに記録する用のリスト
    train_losses = []
    valid_losses = []
    epoch_train_ious = []
    epoch_val_ious = []
    epoch_train_miou = []
    epoch_val_miou = []
    epoch_train_ious_from_confusion_matrix = []
    epoch_val_ious_from_confusion_matrix = []
    epoch_train_miou_from_confusion_matrix = []
    epoch_val_miou_from_confusion_matrix = []
    best_metrics = {"loss": float('inf'), "tumor_iou": 0.0, "mean_iou": 0.0}
    for epoch in range(epochs):
        # 各エポックの損失をリストに追加
        epoch_loss, train_ious, train_confusion_matrix = one_epoch_train(train_loader, device, optimizer, model, criterion, epoch, epochs, class_num)

        # 検証用データ
        val_loss, val_ious, val_confusion_matrix = eval_dataset_and_save_images(
            valid_loader, 
            device, 
            model, 
            criterion,
            class_num,
            segment_dir / f"epoch_{epoch+1}",
            data_dir
            )                     
        train_iou_from_matrix, train_miou_from_matrix = calculate_iou_and_miou_from_confusion_matrix(train_confusion_matrix, class_num)
        val_iou_from_matrix, val_miou_from_matrix = calculate_iou_and_miou_from_confusion_matrix(val_confusion_matrix, class_num)
        tarin_avg_ious, train_miou = calculate_average_ious_and_miou(train_ious)
        val_avg_ious, val_miou = calculate_average_ious_and_miou(val_ious)
        # 最良モデルの保存
        index_mapping = {v: k for k, v in CLASS_MAPPING.items()}
        tumor_index = index_mapping["tumor"]
        tumor_ious_score = val_avg_ious[tumor_index]
        evaluation_metrics  = {
                    "loss": val_loss,
                    "tumor_iou": tumor_ious_score,
                    "mean_iou": val_miou
        }      
        best_metrics = save_best_models(epoch, model, evaluation_metrics, best_metrics, model_save_dir)
        train_losses.append(epoch_loss)
        valid_losses.append(val_loss)
        epoch_train_ious.append(tarin_avg_ious)
        epoch_val_ious.append(val_avg_ious)
        epoch_train_miou.append(train_miou)
        epoch_val_miou.append(val_miou)
        epoch_train_ious_from_confusion_matrix.append(train_iou_from_matrix)
        epoch_val_ious_from_confusion_matrix.append(val_iou_from_matrix)
        epoch_train_miou_from_confusion_matrix.append(train_miou_from_matrix)
        epoch_val_miou_from_confusion_matrix.append(val_miou_from_matrix)
        # logger.info(f'Epoch [{epoch+1}/{epochs}] | Loss: Train {epoch_loss}, Validation {val_loss}\n')

        # 学習曲線をプロット
        plot_and_save_metrics_curve(epoch+1, train_losses, valid_losses, "Loss", graph_save_dir / Path("training_loss_curve.png"))
        plot_and_save_metrics_curve(epoch+1, epoch_train_miou, epoch_val_miou, "MIOU", graph_save_dir / Path("training_miou_curve.png"))
        plot_and_save_metrics_curve(epoch+1, epoch_train_miou_from_confusion_matrix, epoch_val_miou_from_confusion_matrix, "MIOU", graph_save_dir / Path("training_miou_curve_from_confusion_matrix.png"))
        plot_and_save_iou_curve(epoch+1, epoch_train_ious, epoch_val_ious, class_num, graph_save_dir / Path("training_iou_curve.png"))
        plot_and_save_iou_curve(epoch+1, epoch_train_ious_from_confusion_matrix, epoch_val_ious_from_confusion_matrix, class_num, graph_save_dir / Path("training_iou_curve_from_confusion_matrix.png"))

        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break


def one_epoch_train(
    train_loader: DataLoader,   
    device: torch.device,       
    optimizer: optim.Optimizer, 
    model: nn.Module,           
    criterion: nn.Module,       
    epoch: int,                 
    epochs: int,
    class_num                 
) -> float:           
    """
    1エポックごとのモデルの学習を行う関数
    """          
    running_loss = 0.0
    train_ious = []
    train_all_iou_counts = [] #クラスごとのnanの個数
    all_preds = []
    all_targets = []
    model.train()
    for images, masks, _ in train_loader:
        images = images.to(device)

        masks = masks.to(device)

        # 勾配の初期化
        optimizer.zero_grad()

        # モデルの出力
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        logger.debug(f"モデルの出力値のサイズ:{outputs.shape}")
        output_size = outputs.shape[2:]  # 出力の空間サイズ (height, width)
        masks_resized = resize_mask(masks, output_size)  # リサイズする
        logger.debug(f"Output shape: {outputs.shape}, Mask shape: {masks.shape} => MaskResized shape: {masks_resized.shape}")
        # 損失の計算
        loss = criterion(outputs, masks_resized.long())
        
        # 逆伝播と最適化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        #trainのiousの計算
        preds = outputs.argmax(1).cpu().numpy()
        train_iou = calculate_iou(preds, masks_resized, class_num)  # 5クラスの場合
        train_ious, train_all_iou_counts = update_ious_and_counts(train_ious, train_all_iou_counts, train_iou)
        all_preds.extend(preds.flatten())
        all_targets.extend(masks_resized.cpu().numpy().flatten())
    epoch_loss = running_loss / len(train_loader)
    # 混同行列の計算
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=range(class_num))
    logger.debug(f"data_loader_length:{len(train_loader)}")
    logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

    return epoch_loss, train_ious, conf_matrix

def eval_dataset_and_save_images(
    data_loader: DataLoader,   
    device: torch.device,       
    model: nn.Module,           
    criterion: nn.Module,   
    class_num: int, 
    seg_img_dir: Path = None,
    org_img_dir: Path = None
) -> Tuple[float, List[float]] :
    if seg_img_dir is not None:
        seg_img_dir.mkdir(parents=True, exist_ok=True)

    running_loss = 0.
    val_ious = []
    val_all_iou_counts = [] #クラスごとのnanの個数
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for images, masks, image_names in data_loader:
            images, masks = images.to(device), masks.to(device)

            # # 勾配の初期化
            # optimizer.zero_grad()
            # 順伝播
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            # 損失の計算
            output_size = outputs.shape[2:]  # 出力の空間サイズ (height, width)
            masks_resized = resize_mask(masks, output_size)  # リサイズする
            loss = criterion(outputs, masks_resized.long())

            running_loss += loss.item()
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(masks_resized.cpu().numpy().flatten())
            if seg_img_dir is not None:
                # 各画像に対してセグメント化結果を保存
                for i in range(images.size(0)):
                    image_name = image_names[i]
                    output_prediction = preds[i]
                    segment_save(seg_img_dir, org_img_dir / image_name, output_prediction)
            
            # 各クラスごとのIoUを計算
            val_iou = calculate_iou(preds, masks_resized, class_num)  # 5クラスの場合
            val_ious, val_all_iou_counts = update_ious_and_counts(val_ious, val_all_iou_counts, val_iou)
    # 混同行列の計算
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=range(class_num))

    return running_loss / len(data_loader), val_ious, conf_matrix

def parse_args():
    # オプションの解析
    parser = argparse.ArgumentParser(description="脳下垂体腫瘍の検知")

    parser.add_argument("--data_dir",
                        type=Path,
                        default="../../data/",
                        help='入力データのディレクトリパス'
                        )
    parser.add_argument("--mask_dir",
                        type=Path,
                        default="../../data/mask_json",
                        help='入力データのディレクトリパス'
                        )    
    parser.add_argument("--save_dir",
                        type=Path,
                        default="../../result/",
                        help='結果を保存するディレクトリパス'
                        )
    parser.add_argument("--epochs",
                        type=int,
                        default=100
                        )
    parser.add_argument("--model_name",
                        type=str,
                        default="fcn_resnet50",
                        choices=["fcn_resnet50", "fcn_resnet101", "deeplabv3_resnet101", "vit_b_16_segmentation", "fcn_bot_resnet101"],
                        help="Choose the model architecture. Available options are: fcn_resnet50, fcn_resnet101, fcn_bot_resnet101, deeplabv3_resnet101, vit_b_16_segmentation."
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=20
                        )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4
                        )
    parser.add_argument("--patience",
                        type=int,
                        default=5
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
    parser.add_argument('--attention_mode', 
                        type=str,
                        default="none",
                        choices=["none", "self_attention", "channel_attention", "both"],
                        help='attention_layerの使用するかを指定する変数'
                        )
    parser.add_argument("--seed",
                        type=float,
                        default=42
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
    #再現性確保のためのseed値固定
    seed_everything(args.seed)
    logger.setLevel(args.loglevel.upper())
    logger.info("loglevel: %s", args.loglevel)
    lformat = "%(name)s <L%(lineno)s> [%(levelname)s] %(message)s"
    logging.basicConfig(
        # filename='../../train_fcn.log',  # 出力先ファイルを指定
        # stream=sys.stdout,  # 標準出力に出力
        level=logging.INFO,
        filemode='w',  # ファイルを上書きモードに設定
        format=lformat,
    )
    logger.setLevel(args.loglevel.upper())


    main(args)
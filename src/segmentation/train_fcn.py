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
from typing import Tuple, List

sys.path.append("../")
# 自作モジュール

#データセット関連
from segment_utils.dataset_utils import(
    transform,
    SegmentationDataset,
    split_dataset
)
# 画像処理関連
from segment_utils.image_processing import(
    segment_save
)
# 評価関連
from segment_utils.metrics import(
    save_iou_to_csv_from_conf_matrix,
    save_confusion_matrix_with_metrics
)
# グラフ
from segment_utils.graph import(
    plot_and_save_learning_curve,
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
    metrics_save_dir = save_dir / Path("metrics")
    others_save_dir = save_dir / Path("others")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(graph_save_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)
    class_names = ['background', 'sellar', 'sella', 'pituitary', 'tumor']
    """モデルをデバイス（GPU/CPU）に設定し、必要に応じてマルチGPUモードに切り替えます。"""

    # COCOデータセットで事前学習されたモデルをロード
    model = setup_fcn_model(model_name, attention_mode, num_classes=class_num)
    # デバイスの設定（GPUが利用可能なら使用）
    device, model = setup_device(model)

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
    train_model(train_loader, valid_loader, device, optimizer, model, criterion, epochs, early_stopping, class_num, class_names, train_image_dir, graph_save_dir, model_save_dir, metrics_save_dir)

    return

def save_best_model(
    model: nn.Module, 
    epoch: int, 
    epoch_loss: float, 
    best_loss: float, 
    model_save_dir: Path
) -> float:
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        model_save_path = model_save_dir / Path("best_model_segment.pth")
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Best model saved with loss {best_loss:.4f} at epoch {epoch+1}")
    return best_loss

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
    metrics_save_dir: Path               
) -> None:                  
    """"""  
    logger.info("学習を開始します。")  
    # 損失,iousをエポックごとに記録する用のリスト
    train_losses = []
    valid_losses = []
    epoch_ious = []
    best_loss = float('inf')  
    for epoch in range(epochs):
        # 各エポックの損失をリストに追加
        epoch_loss = one_epoch_train(train_loader, device, optimizer, model, criterion, epoch, epochs)

        # 検証用データ
        val_loss, ious = eval_dataset_and_save_images(
            valid_loader, 
            device, 
            model, 
            criterion,
            class_num,
            graph_save_dir / f"epoch_{epoch+1}",
            data_dir
            )

        # 最良モデルの保存
        best_loss = save_best_model(model, epoch, val_loss, best_loss, model_save_dir)

        train_losses.append(epoch_loss)
        valid_losses.append(val_loss)
        
        mean_iou = torch.tensor(ious).mean(dim=0).tolist()  # クラスごとの平均
        epoch_ious.append(mean_iou)
        # logger.info(f'Epoch [{epoch+1}/{epochs}] | Loss: Train {epoch_loss}, Validation {val_loss}\n')

        # 学習曲線をプロット
        plot_and_save_learning_curve(epoch+1, train_losses, valid_losses, graph_save_dir)
        plot_and_save_iou_curve(epoch+1, epoch_ious, class_num, graph_save_dir)

        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    save_metrics(model, train_loader, device, class_num, class_names, metrics_save_dir)
    save_metrics(model, valid_loader, device, class_num, class_names, metrics_save_dir)


def one_epoch_train(
    train_loader: DataLoader,   
    device: torch.device,       
    optimizer: optim.Optimizer, 
    model: nn.Module,           
    criterion: nn.Module,       
    epoch: int,                 
    epochs: int                 
) -> float:           
    """
    1エポックごとのモデルの学習を行う関数
    """          
    running_loss = 0.0
    model.train()
    for images, masks, _ in train_loader:
        images = images.to(device)

        masks = masks.to(device)

        # 勾配の初期化
        optimizer.zero_grad()

        # モデルの出力
        outputs = model(images)['out']
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
    epoch_loss = running_loss / len(train_loader)
    logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

    return epoch_loss

def eval_dataset_and_save_images(
    data_loader: DataLoader,   
    device: torch.device,       
    model: nn.Module,           
    criterion: nn.Module,   
    class_num: int, 
    seg_img_dir: Path = None,
    org_img_dir: Path = None
) -> float :
    if seg_img_dir is not None:
        seg_img_dir.mkdir(parents=True, exist_ok=True)

    running_loss = 0.
    ious = []
    model.eval()
    with torch.no_grad():
        for images, masks, image_names in data_loader:
            images, masks = images.to(device), masks.to(device)

            # # 勾配の初期化
            # optimizer.zero_grad()
            # 順伝播
            outputs = model(images)['out']
            
            # 損失の計算
            output_size = outputs.shape[2:]  # 出力の空間サイズ (height, width)
            masks_resized = resize_mask(masks, output_size)  # リサイズする
            loss = criterion(outputs, masks_resized.long())

            running_loss += loss.item()

            if seg_img_dir is not None:
                preds = outputs.argmax(1).cpu().numpy()
                # 各画像に対してセグメント化結果を保存
                for i in range(images.size(0)):
                    image_name = image_names[i]
                    output_prediction = preds[i]
                    segment_save(seg_img_dir, org_img_dir / image_name, output_prediction)
            
            # 各クラスごとのIoUを計算
            iou = calculate_iou(preds, masks, class_num)  # 5クラスの場合
            ious.append(iou)

    return running_loss / len(data_loader), ious

# IoUを計算する関数
def calculate_iou(
    pred: np.ndarray, 
    target: torch.Tensor, 
    num_classes: int
) -> List[float]:
    """
    各クラスの通常のIoU計算。優先度に基づかないピクセルの評価。
    """
    # predがNumPy配列の場合、テンソルに変換
    pred = torch.tensor(pred)

    # predをtargetと同じデバイスに移動
    pred = pred.to(target.device)

    # predの形状がtargetと異なる場合、リサイズ
    if pred.shape != target.shape:
        pred = F.interpolate(pred.unsqueeze(0).float(), size=target.shape[-2:], mode='nearest').squeeze(0)

    # predとtargetの形状を1次元に変換
    pred = pred.view(-1)
    target = target.view(-1)

    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        
        # IoU計算
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()
        
        if union == 0:
            ious.append(float('nan'))  # クラスが存在しなければNaNを追加
        else:
            ious.append(intersection / union)

    return ious


def save_metrics(model, test_loader, device, num_classes, class_names, save_path):
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, masks, _ in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            # モデルの出力
            outputs = model(images)['out']
            output_resized = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            output_predictions = output_resized.argmax(1).squeeze().cpu().numpy()

            # ピクセル単位で予測ラベルと正解ラベルをフラットにする
            all_predictions.append(output_predictions.flatten())
            all_ground_truths.append(masks.cpu().numpy().flatten())

    # すべての画像の正解ラベルと予測ラベルをまとめる
    all_predictions = np.concatenate(all_predictions)
    all_ground_truths = np.concatenate(all_ground_truths)

    # 混同行列の作成
    conf_matrix = confusion_matrix(all_ground_truths, all_predictions, labels=list(range(num_classes)))

    # 混同行列とIoUを保存
    save_confusion_matrix_with_metrics(conf_matrix, save_path, class_names)
    save_iou_to_csv_from_conf_matrix(conf_matrix, class_names, save_path, num_classes)

    logger.info("IoU and Confusion Matrix saved.")

def resize_mask(
    mask: torch.Tensor,
    output_size: Tuple[int, int]  # モデルの出力サイズを引数に追加
) -> torch.Tensor:
    return F.interpolate(mask.unsqueeze(1).float(), size=output_size, mode='nearest').squeeze(1).long()

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
                        choices=["fcn_resnet50", "fcn_resnet101", "fcn_vgg16", "fcn_vgg19", "deeplabv3_resnet101"],
                        help="Choose the model architecture. Available options are: fcn_resnet50, fcn_resnet101, fcn_vgg16, fcn_vgg19, deeplabv3_resnet101."
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
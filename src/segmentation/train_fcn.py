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
from typing import Tuple, List

sys.path.append("../")
# 自作モジュール
from segmentation import (
    transform, 
    SegmentationDataset,
    segment_save,
    visualize_random_sample_from_dataset
)

from preprocessing.split_dataset import (
    split_dataset
)

from utils.model_utils import (
    setup_fcn_model,
    setup_device,
    EarlyStopping
)
from utils.set_seed import (
    seed_everything
)

logger = logging.getLogger(__name__)



def main(args):
    data_dir = args.data_dir
    mask_dir = args.mask_dir
    model_name = args.model_name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir / Path("nagoya", "training_results", model_name, str(args.batch_size), "{:.1e}".format(args.learning_rate))
    epochs = args.epochs
    class_num = args.class_num
    train_image_dir = data_dir / Path("img")
    model_save_dir = save_dir / Path( "model")
    graph_save_dir = save_dir / Path("graph")
    others_save_dir = save_dir / Path("others")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(graph_save_dir, exist_ok=True)

    #モデルの学習の際に重複時の優先度を決定する
    priority = [4, 3, 2, 1, 0]
    """モデルをデバイス（GPU/CPU）に設定し、必要に応じてマルチGPUモードに切り替えます。"""

    # COCOデータセットで事前学習されたFCN-ResNet50モデルをロード
    model = setup_fcn_model(model_name, num_classes=class_num)
    # デバイスの設定（GPUが利用可能なら使用）
    device, model = setup_device(model)

    # データセットとデータローダの作成
    dataset = SegmentationDataset(mask_dir, train_image_dir, transform)
    logger.debug(f"Total samples in dataset: {len(dataset)}")

    # サンプルを取得して確認
    sample_idx = 0  # 確認したいインデックス
    image, mask, filename = dataset[sample_idx]  # 0番目のサンプルを取得

    if image is None or mask is None:
        logger.debug(f"Sample {sample_idx} has no valid data.")
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset, output_dir=others_save_dir / "split_dataset")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=True, pin_memory=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=True, pin_memory=True, num_workers=0)

    # マスクのラベルが指定したクラス数に収まっているか確認する例
    mask = train_dataset[10][1]  # 0番目のサンプルのマスクを取得
    # mask = map_mask_to_four_classes(mask)  # クラスのマッピングを実行
    logger.info(f'Categories after mapping: {mask.unique()}')  # マスク内のユニークなクラスラベルを確認


    # 実行時のモデルアーキテクチャとデータの可視化
    save_model_architecture(model, others_save_dir)
    visualize_random_sample_from_dataset(train_dataset, others_save_dir)

    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()  # セマンティックセグメンテーションの損失関数
    optimizer = optim.Adam(model.parameters(), lr=args.lerning_rate)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True) 

    #モデルの学習と学習曲線の出力
    train_model(train_loader, valid_loader, device, optimizer, model, criterion, epochs, early_stopping, class_num, priority, train_image_dir, graph_save_dir, model_save_dir)

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
    priority: List[int],
    data_dir: Path,
    graph_save_dir: Path,
    model_save_dir: Path                 
) -> None:                    
    logger.info("学習を開始します。")  
    # 損失を記録するリスト
    train_losses = []
    valid_losses = []
    # 各エポックごとのIoUを格納するリスト
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
            priority,
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


def one_epoch_train(
    train_loader: DataLoader,   
    device: torch.device,       
    optimizer: optim.Optimizer, 
    model: nn.Module,           
    criterion: nn.Module,       
    epoch: int,                 
    epochs: int                 
) -> float:                     
    running_loss = 0.0
    model.train()
    for images, masks, _ in train_loader:
        images = images.to(device)

        masks = masks.to(device)

        # 勾配の初期化
        optimizer.zero_grad()

        # 順伝播
        outputs = model(images)['out']
        logger.debug(f"Output shape: {outputs.shape}, Mask shape: {masks.shape}")
        masks_resized = resize_mask(masks)
        logger.debug(f"Output shape: {outputs.shape}, MaskResized shape: {masks_resized.shape}")
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
    priority: List[int],
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
            masks_resized = resize_mask(masks)
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
            iou = calculate_priority_based_iou(preds, masks, class_num, priority)  # 5クラスの場合
            ious.append(iou)

    return running_loss / len(data_loader), ious

# モデルアーキテクチャを保存する関数
def save_model_architecture(model: torch.nn.Module, save_path: Path):
    # モデルアーキテクチャを文字列化
    model_str = str(model)

    # ファイルに保存
    save_path = save_path / "model_architecture.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)  # ディレクトリが存在しない場合、作成する
    with open(save_path, 'w') as f:
        f.write(model_str)
    
    return

def plot_and_save_learning_curve(
    epochs: int, 
    train_losses: float, 
    valid_losses: float, 
    graph_save_dir: Path
) -> None:
    # プロットのリセット
    plt.figure()  # 新しい図を作成
    plt.clf()     # 既存の図をクリア
    
    # 学習曲線をプロット
    plt.plot(range(1, epochs + 1), train_losses, label="Training")
    plt.plot(range(1, epochs + 1), valid_losses, label="Validation")
    
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Learnin Loss Curve")
    plt.legend()
    
    # 学習曲線をファイルに保存
    plt.savefig(graph_save_dir / Path("training_loss_curve.png"))

def plot_and_save_iou_curve(
    epochs: int,
    epoch_ious: List[List[float]],  # 各クラスごとのIoU値
    num_classes: int,               # クラス数
    graph_save_dir: Path
) -> None:
    # IoU曲線のプロット
    plt.figure()  # 新しい図を作成
    plt.clf()     # 既存の図をクリア

    # epoch_iousは [エポック数][クラス数] の形のリストを仮定
    epoch_ious = torch.tensor(epoch_ious)  # 形を整えるためにテンソルに変換

    # 各クラスごとにIoUの変化をプロット
    for cls in range(num_classes):
        plt.plot(range(1, epochs + 1), epoch_ious[:, cls], label=f'Class {cls} IoU')

    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU per Epoch for Each Class")
    plt.legend()
    plt.grid(True)

    # IoU曲線をファイルに保存
    plt.savefig(graph_save_dir / Path("iou_per_epoch_curve.png"))
    plt.close()

# IoUを計算する関数
def calculate_priority_based_iou(
    pred: np.ndarray, 
    target: torch.Tensor, 
    num_classes: int, 
    priority: List[int]  # priorityはクラスIDなのでint型に修正
) -> List[float]:
    """
    優先度に基づくIoU計算。重複しているピクセルを優先度に基づいて適切に評価。
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
        
        # 優先順位に基づく調整
        if cls in priority:
            intersection = (pred_inds & target_inds).sum().float().item()
            union = (pred_inds | target_inds).sum().float().item()
            if union == 0:
                ious.append(float('nan'))  # クラスが存在しなければNaNを追加
            else:
                ious.append(intersection / union)
        else:
            ious.append(float('nan'))  # クラスが存在しなければNaNを追加

    return ious



def resize_mask(
    mask: torch.Tensor
) -> torch.Tensor:
    return F.interpolate(mask.unsqueeze(1).float(), size=(520, 520), mode='nearest').squeeze(1).long()

def parse_args():
    # オプションの解析
    parser = argparse.ArgumentParser(description="骨格データの生成")

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
                        default="fcn_resnet50"
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=20
                        )
    parser.add_argument("--lerning_rate",
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
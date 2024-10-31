import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List


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
    plt.close()  # 図を閉じる

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
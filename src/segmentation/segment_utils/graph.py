import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List


def plot_and_save_metrics_curve(
    epochs: int, 
    train_losses: float, 
    valid_losses: float, 
    title_label: str,
    graph_path: Path
) -> None:
    # プロットのリセット
    plt.figure()  # 新しい図を作成
    plt.clf()     # 既存の図をクリア
    
    # 学習曲線をプロット
    plt.plot(range(1, epochs + 1), train_losses, label="Training")
    plt.plot(range(1, epochs + 1), valid_losses, label="Validation")
    
    plt.xlabel("Epoch")
    plt.ylabel(f"{title_label}")
    plt.title(f"Learnin {title_label} Curve")
    plt.legend()
    
    # 学習曲線をファイルに保存
    plt.savefig(graph_path)
    plt.close()  # 図を閉じる

def plot_and_save_iou_curve(
    epochs: int,
    epoch_train_ious: List[List[float]],
    epoch_val_ious: List[List[float]],
    num_classes: int,
    graph_path: Path
) -> None:
    # IoU曲線のプロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # 2つのサブプロットを横に並べる

    epoch_train_ious = torch.tensor(epoch_train_ious)
    epoch_val_ious = torch.tensor(epoch_val_ious)

    # Training IoU
    for cls in range(num_classes):
        ax1.plot(range(1, epochs + 1), epoch_train_ious[:, cls], label=f'Class {cls} IoU')
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("IoU")
    ax1.set_title("Training IoU per Epoch for Each Class")
    ax1.legend()
    ax1.grid(True)

    # Validation IoU
    for cls in range(num_classes):
        ax2.plot(range(1, epochs + 1), epoch_val_ious[:, cls], label=f'Class {cls} IoU')
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("Validation IoU per Epoch for Each Class")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()  # サブプロット間のスペースを自動調整

    # IoU曲線をファイルに保存
    plt.savefig(graph_path)
    plt.close()
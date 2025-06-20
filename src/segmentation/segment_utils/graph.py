import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List, Dict
import numpy as np

def plot_and_save_metrics_curve(
    epochs: int, 
    train_losses: List[float], 
    valid_losses: List[float], 
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

def plot_iou_by_image(
    image_iou_dict: Dict[str, List[float]], 
    class_names: List[str], 
    save_dir: Path
) -> None:
    """
    画像ごと、クラスごとのIoU値をプロット。
    X軸は画像１枚ごとに１増えていくが、ラベルは5刻みに整理。
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # 画像名とクラスごとのIoU値を整理
    image_names = list(image_iou_dict.keys())  # 画像数
    iou_values = np.array([list(image_iou_dict[name]) for name in image_names])

    # 3クラス用の鮮やかな色定義
    distinct_colors = {
        'sella': '#FF5733',      # オレンジ
        'pituitary': '#33A1FD',  # 青
        'tumor': '#4CAF50'       # 緑
    }
    
    # プロット領域の設定
    plt.figure(figsize=(15, 8))
    max_idx = len(image_names) - 1
    
    # 各クラスについてプロット - すべて円形マーカー
    for class_idx, class_name in enumerate(class_names):
        color = distinct_colors.get(class_name, "#000000")
        plt.plot(range(len(image_names)), iou_values[:, class_idx],
                 marker='o', markersize=8, linewidth=2.5,
                 label=class_name, color=color)

    # X軸に5刻みのラベルを設定（画像1枚＝1）
    tick_indices = range(0, max_idx + 1, 5)  # 5刻み
    tick_labels = [str(i) for i in tick_indices]
    plt.xticks(tick_indices, tick_labels, rotation='horizontal', ha='center', fontsize=10)

    # ラベel設定
    plt.xlabel('Time', fontsize=12, fontweight='bold')
    plt.ylabel('IoU', fontsize=12, fontweight='bold')

    # 凡例整理
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12, frameon=True, fancybox=True, shadow=True)

    # グリッドとY軸設定
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)  # IoUは0から1の範囲
    plt.xlim(-1, max_idx + 1)

    # 余白整理して保存
    plt.tight_layout()
    save_file = save_dir / 'iou_by_image.png'
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close()

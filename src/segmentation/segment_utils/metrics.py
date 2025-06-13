from pathlib import Path
import numpy as np
import csv
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def calculate_iou_and_miou_from_confusion_matrix(
    conf_matrix: np.ndarray, 
    num_classes: int
) -> Tuple[List[float], float]:
    """
    混同行列から各クラスのIoUとmIoUを計算する関数。
    """
    iou_per_class = []

    for i in range(num_classes):
        tp = conf_matrix[i, i]  # True Positive (TP)
        fp = conf_matrix[:, i].sum() - tp  # False Positive (FP)
        fn = conf_matrix[i, :].sum() - tp  # False Negative (FN)

        # IoU = TP / (TP + FP + FN)
        denom = tp + fp + fn
        if denom > 0:
            iou = tp / denom
        else:
            iou = float('nan')  # クラスが存在しない場合はNaNにする
        iou_per_class.append(iou)
    
    # mIoUの計算
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else float('nan')
    
    return iou_per_class, miou

def save_iou_to_csv(
    iou_per_class: List[float],
    miou: np.ndarray, 
    class_names: List[str], 
    save_path: Path, 
    file_name: Path,
    num_classes: int
) -> None:
    """
    混同行列から計算したIoUをCSVに保存し、最後にmIoUを記載する関数。
    """
    # CSVファイルの保存先を指定
    csv_file = save_path / file_name

    # CSVファイルが存在しない場合はヘッダーを作成
    if not csv_file.exists():
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # IoUのみを表示するためのヘッダー
            writer.writerow(['Class', 'IoU'])

    # IoU結果をCSVに書き込む (1行目はクラス名、2行目以降にIoUを出力)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(num_classes):
            # クラス名とIoUをそれぞれの行に書き込み
            writer.writerow([class_names[i], round(iou_per_class[i], 3)])

        # 最後の行にmIoUを書き込む
        writer.writerow(['mIoU', round(miou, 3)])

    print(f"IoU and mIoU results saved to {csv_file}")

def save_confusion_matrix_with_metrics(
    conf_matrix: np.ndarray, 
    save_path: Path, 
    class_names: List[str]
) -> None:
    # 保存するファイルのパスを設定
    confusion_matrix_file = save_path / "confusion_matrix_with_metrics.csv"
    num_classes = len(class_names)
    with open(confusion_matrix_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # ヘッダー (クラス名を横に並べる)
        header = [''] + class_names + ['Precision']
        writer.writerow(header)

        precision_list = []
        recall_list = []
        total_tp = total_fp = total_fn = total_tn = 0

        # 各クラスごとの混同行列の情報と適合率を出力
        for i in range(num_classes):
            tp = conf_matrix[i, i]
            fn = conf_matrix[i, :].sum() - tp
            fp = conf_matrix[:, i].sum() - tp
            tn = conf_matrix.sum() - (tp + fn + fp)

            # 適合率（Precision） = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_list.append(round(precision, 3))

            # 再現率（Recall） = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_list.append(round(recall, 3))

            # 各クラスの混同行列と適合率を出力
            row = [class_names[i]] + list(conf_matrix[i, :]) + [round(precision, 3)]
            writer.writerow(row)

            # 合計値を計算（Accuracy用）
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        # 最後の行に再現率を追加
        writer.writerow(['Recall'] + recall_list + [''])  # Recall行の出力, 空白を追加してずれを防ぐ

        # 全体の精度 = (TP + TN) / (TP + TN + FP + FN)
        total_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        writer.writerow([''] * 6 + [round(total_accuracy, 3)])

    print(f"Confusion matrix with metrics saved to: {confusion_matrix_file}")

def calculate_iou(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    num_classes: int
) -> Tuple[List[float], List[int]]:
    """
    各クラスのIoUを計算し、NaNの数も追跡します。
    """
    # predがNumPy配列の場合、テンソルに変換
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)

    # predをtargetと同じデバイスに移動
    pred = pred.to(target.device)

    # サイズチェック（オプション）
    assert pred.shape == target.shape, "Prediction and target shapes must match"

    # predとtargetの形状を1次元に変換
    pred = pred.view(-1)
    target = target.view(-1)

    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()
        target_sum = target_inds.sum().float().item()
        
        if union == 0:
            if intersection == 0:
                ious.append(float('nan'))  # どちらにも存在しない
            else:
                raise ValueError(f"Unexpected case: intersection > 0 but union == 0 for class {cls}")
        elif intersection == 0:
            if target_sum > 0:
                ious.append(0.0)  # 正解にしか存在しない
            else:
                ious.append(float('nan'))  # 予測にしか存在しない
        else:
            ious.append(intersection / union)
    
    return ious

def calculate_target_class_iou(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> List[float]:
    """
    修正版のIoU計算関数。
    以下の3クラスのIoUを計算する:
    - sella（元の1,2,3,4のすべてをマージ）
    - pituitary（元の3）
    - tumor（元の4）
    """
    # predがNumPy配列の場合、テンソルに変換
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)

    # predをtargetと同じデバイスに移動
    pred = pred.to(target.device)

    # サイズチェック
    assert pred.shape == target.shape, "Prediction and target shapes must match"

    # 形状を1次元に変換
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    ious = []
    
    # 1. sellaのIoU計算（元の1,2,3,4を全て含む）
    pred_sella = (pred_flat == 1) | (pred_flat == 2) | (pred_flat == 3) | (pred_flat == 4)
    target_sella = (target_flat == 1) | (target_flat == 2) | (target_flat == 3) | (target_flat == 4)
    
    intersection_sella = (pred_sella & target_sella).sum().float().item()
    union_sella = (pred_sella | target_sella).sum().float().item()
    
    if union_sella == 0:
        if intersection_sella == 0:
            ious.append(float('nan'))  # どちらにも存在しない
        else:
            raise ValueError("Unexpected case: intersection > 0 but union == 0 for sella")
    else:
        iou_sella = intersection_sella / union_sella
        ious.append(iou_sella)
    
    # 2. pituitaryのIoU計算（元の3）
    pred_pituitary = (pred_flat == 3)
    target_pituitary = (target_flat == 3)
    
    intersection_pituitary = (pred_pituitary & target_pituitary).sum().float().item()
    union_pituitary = (pred_pituitary | target_pituitary).sum().float().item()
    
    if union_pituitary == 0:
        if intersection_pituitary == 0:
            ious.append(float('nan'))  # どちらにも存在しない
        else:
            raise ValueError("Unexpected case: intersection > 0 but union == 0 for pituitary")
    else:
        iou_pituitary = intersection_pituitary / union_pituitary
        ious.append(iou_pituitary)
    
    # 3. tumorのIoU計算（元の4）
    pred_tumor = (pred_flat == 4)
    target_tumor = (target_flat == 4)
    
    intersection_tumor = (pred_tumor & target_tumor).sum().float().item()
    union_tumor = (pred_tumor | target_tumor).sum().float().item()
    
    if union_tumor == 0:
        if intersection_tumor == 0:
            ious.append(float('nan'))  # どちらにも存在しない
        else:
            raise ValueError("Unexpected case: intersection > 0 but union == 0 for tumor")
    else:
        iou_tumor = intersection_tumor / union_tumor
        ious.append(iou_tumor)
    
    return ious

def update_ious_and_counts(
    all_ious: List[List[float]],
    all_iou_counts: List[int],
    ious: List[float]
) -> Tuple[List[float], List[int]]:
    """
    IoUリストとカウントを更新する関数
    """
    if not all_ious:
        updated_ious = [[x] if not math.isnan(x) else None for x in ious]
        updated_counts = [1 if not math.isnan(x) else 0 for x in ious]
    else:
        updated_ious = [
            x + [y] if x is not None and not math.isnan(y) else
            [y] if x is None and not math.isnan(y) else
            x for x, y in zip(all_ious, ious)
        ]
        updated_counts = [count + (1 if not math.isnan(y) else 0) for count, y in zip(all_iou_counts, ious)]
    
    return updated_ious, updated_counts

def calculate_average_ious_and_miou(
    all_ious: List[Optional[List[float]]]
) -> tuple[List[float], float]:
    """
    クラスごとの平均IoUとmIoUを計算する関数
    """
    # クラスごとの平均IoUを計算
    avg_ious = [
        np.mean(iou_list) if iou_list is not None else float('nan')
        for iou_list in all_ious
    ]

    # NaNでない有効なIoUのみを抽出
    valid_ious = [iou for iou in avg_ious if not math.isnan(iou)]

    # mIoUを計算（有効なIoUがある場合のみ）
    miou = np.mean(valid_ious) if valid_ious else float('nan')

    return avg_ious, miou

def plot_iou_by_image(
    image_iou_dict: dict, 
    class_names: List[str], 
    save_dir: Path
) -> None:
    """
    画像ごと、クラスごとのIoU値をプロット。
    """
    plt.figure(figsize=(15, 8))
    
    # 画像名とクラスごとのIoU値を整理
    image_names = list(image_iou_dict.keys())
    iou_values = np.array([list(image_iou_dict[name]) for name in image_names])
    
    # 3クラス用の鮮やかな色定義
    distinct_colors = {
        'sella': '#FF5733',      # オレンジ
        'pituitary': '#33A1FD',  # 青
        'tumor': '#4CAF50'       # 緑
    }
    
    # 各クラスについてプロット - すべて円形マーカー
    for class_idx, class_name in enumerate(class_names):
        plt.plot(range(len(image_names)), iou_values[:, class_idx], 
                 marker='o', markersize=8, linewidth=2.5,
                 label=class_name, color=distinct_colors[class_name])
    
    # グラフの設定
    plt.xticks(range(len(image_names)), image_names, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Image Name', fontsize=12, fontweight='bold')
    plt.ylabel('IoU', fontsize=12, fontweight='bold')
    plt.title('IoU by Image and Class', fontsize=14, fontweight='bold')
    
    # 凡例を見やすく配置
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12, frameon=True, 
               fancybox=True, shadow=True)
    
    # グリッドとY軸の範囲設定
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)  # IoUは0から1の範囲
    
    # 余白を調整
    plt.tight_layout()
    
    # グラフを保存
    plt.savefig(save_dir / 'iou_by_image.png', bbox_inches='tight', dpi=300)
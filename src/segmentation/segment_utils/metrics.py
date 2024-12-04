from pathlib import Path
import numpy as np
import csv
import torch
import torch.nn.functional as F
from typing import List

def calculate_iou_from_confusion_matrix(
    conf_matrix: np.ndarray, 
    num_classes: int
) -> List[float]:
    """
    混同行列から各クラスのIoUを計算する関数。

    Args:
        conf_matrix (numpy.ndarray): 混同行列
        num_classes (int): クラス数

    Returns:
        iou_per_class (list): 各クラスのIoUのリスト
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
    
    return iou_per_class


def save_iou_to_csv_from_conf_matrix(
    conf_matrix: np.ndarray, 
    class_names: list, 
    save_path: Path, 
    num_classes: int
) -> None:
    """
    混同行列から計算したIoUをCSVに保存し、最後にmIoUを記載する関数。

    Args:
        conf_matrix (numpy.ndarray): 混同行列
        class_names (list): クラス名のリスト
        save_path (Path): 保存先のパス
        num_classes (int): クラス数。デフォルトは5。
    """
    # IoUを計算
    iou_per_class = calculate_iou_from_confusion_matrix(conf_matrix, num_classes)
    # mIoU (mean IoU) を計算
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]  # 有効なIoUのみ
    miou = np.mean(valid_ious) if valid_ious else float('nan')

    # CSVファイルの保存先を指定
    csv_file = save_path / "iou_results_from_conf_matrix.csv"

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
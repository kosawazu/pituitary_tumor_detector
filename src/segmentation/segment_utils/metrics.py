from pathlib import Path
import numpy as np
import csv
import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

def calculate_iou_and_miou_from_confusion_matrix(
    conf_matrix: np.ndarray, 
    num_classes: int
) -> Tuple[List[float], float]:
    """
    混同行列から各クラスのIoUとmIoUを計算する関数。

    Args:
        conf_matrix (numpy.ndarray): 混同行列
        num_classes (int): クラス数

    Returns:
        Tuple[List[float], float]: 各クラスのIoUのリストとmIoU
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

    Args:
        conf_matrix (numpy.ndarray): 混同行列
        class_names (list): クラス名のリスト
        save_path (Path): 保存先のパス
        num_classes (int): クラス数。デフォルトは5。
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

def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Tuple[List[float], List[int]]:
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
        pred_sum = pred_inds.sum().float().item()
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


def update_ious_and_counts(all_ious, all_iou_counts, ious):
    """
    IoUリストとカウントを更新する関数

    Parameters:
    all_ious (List[List[float] | None]): これまでのIoUのリスト
    all_iou_counts (List[int]): これまでの有効なIoUのカウント
    ious (List[float]): 新しいIoUのリスト

    Returns:
    Tuple[List[List[float] | None], List[int]]: 更新されたIoUリストとカウント
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

    Parameters:
    all_ious (List[Optional[List[float]]]): 各クラスのIoUリスト。Noneの場合はそのクラスに有効なIoUがないことを示す。

    Returns:
    Tuple[List[float], float]: 
        - クラスごとの平均IoU（NaNを含む可能性あり）
        - mIoU（全クラスの有効な平均IoUの平均）
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
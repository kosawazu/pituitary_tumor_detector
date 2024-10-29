from pathlib import Path
import random
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt
import csv
from typing import Union, List

# fcn_resnet50用
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

colors = {
            0: (0, 0, 255),      # 背景 - 青
            1: (0, 255, 0),      # sellar - 緑
            2: (255, 0, 0),      # sella - 赤
            3: (255, 255, 0),    # pituitary - 黄
            4: (128, 0, 128)     # tumor - 紫
        }
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_dir: Path, image_dir: Path, transform=None):
        """
        Args:
            annotation_dir (Path): アノテーションファイルが含まれるディレクトリ
            image_dir (Path): 画像ファイルのディレクトリ
            transform (callable, optional): 画像に対して適用するトランスフォーム
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # ディレクトリ内のすべてのJSONファイルを取得
        self.annotation_files = list(Path(annotation_dir).glob('*.json'))

    def __len__(self):
        return len(self.annotation_files)  # JSONファイルの数に基づいてデータセットの長さを決定

    def __getitem__(self, idx):
        # 該当するアノテーションファイルを読み込む
        annotation_file = self.annotation_files[idx]
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # 画像ファイル名を取得
        img_metadata = data['asset']
        img_filename = img_metadata['name']  # ファイル名のみ取得

        # 画像のフルパスを作成 (image_dir からファイル名を探す)
        img_path = self.image_dir / img_filename

        # 画像の存在を確認、存在しない場合はスキップ
        if not img_path.exists():
            print(f"Image not found: {img_path}, skipping.")
            return None, None, None  # None を返すのではなく例外を処理する

        # 画像を読み込む
        try:
            image = Image.open(img_path).convert('RGB')  # 画像はPIL形式で読み込む
        except FileNotFoundError:
            print(f"Image not found: {img_path}, skipping.")
            return None, None, None

        # アノテーション領域を読み込む
        mask = np.zeros((img_metadata['size']['height'], img_metadata['size']['width']), dtype=np.uint8)

        # アノテーション情報からマスクを作成
        regions = data.get('regions', [])
        for region in regions:
            points = region['points']
            polygon = [(point['x'], point['y']) for point in points]

            # ポリゴンをバイナリマスクに変換
            img_mask = Image.new('L', (img_metadata['size']['width'], img_metadata['size']['height']), 0)
            ImageDraw.Draw(img_mask).polygon(polygon, outline=1, fill=1)
            region_mask = np.array(img_mask)

            # カテゴリごとにマスクを作成（カテゴリ名で対応付け）
            #領域が重複している場合はIDが大きい方が処理として優先される。
            if "sellar" in region['tags']:
                mask = np.maximum(mask, region_mask * 1)  # クラスID 1を使用
            elif "sella" in region['tags']:
                mask = np.maximum(mask, region_mask * 2)  # クラスID 2を使用
            elif "pituitary" in region['tags']:
                mask = np.maximum(mask, region_mask * 3)  # クラスID 3を使用
            elif "tumor" in region['tags']:
                mask = np.maximum(mask, region_mask * 4)  # クラスID 4を使用

        # トランスフォームを適用（もし指定されていれば）
        if self.transform is not None and isinstance(image, Image.Image):  # PIL.Imageのときのみトランスフォームを適用
            image = self.transform(image)  # 画像にリサイズ等のトランスフォームを適用

        # マスクもTensorに変換
        mask = torch.as_tensor(mask, dtype=torch.int64)

        return image, mask, img_filename


    def get_mean_std(self):
        """
        データセットのtransformに設定されたNormalizeのmeanとstdを返すメソッド
        """
        if self.transform is not None:
            for t in self.transform.transforms:  # Compose内のtransformsリストを確認
                if isinstance(t, transforms.Normalize):
                    return t.mean, t.std
        return None, None  # Normalizeが見つからない場合

def segment_save(graph_save_dir, image_path, output_predictions):
    # 元画像の読み込み
    img = Image.open(image_path).convert('RGB')

    # output_predictions がPyTorchテンソルの場合、NumPy配列に変換
    if isinstance(output_predictions, torch.Tensor):
        output_predictions = output_predictions.squeeze().byte().cpu().numpy()

    # output_predictions が NumPy配列の場合
    if isinstance(output_predictions, np.ndarray):
        # ここでデータ型と形状を変換
        # uint8に変換して、0-255の範囲にスケールする
        if output_predictions.dtype != np.uint8:
            output_predictions = output_predictions.astype(np.uint8)
        
        # カラーマッピングの設定（背景: 青, 紙袋: 緑, 傷: 赤, クラス4: 黄, クラス5: 紫）

        # カラーマップに基づいて output_predictions を色付け
        output_colored = np.zeros((*output_predictions.shape, 3), dtype=np.uint8)
        for class_index, color in colors.items():
            output_colored[output_predictions == class_index] = color

        # PIL画像に変換
        output_image = Image.fromarray(output_colored)

    # セグメンテーション結果のサイズを元画像に合わせる
    output_image_resized = output_image.resize(img.size, resample=Image.NEAREST)

    # プロットの設定（2つのサブプロットを横並び）
    _, ax = plt.subplots(1, 2, figsize=(20, 10))

    # 元画像を表示
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')  # 軸を非表示

    # セグメンテーション結果を表示
    ax[1].imshow(output_image_resized)  # カラーマップを使ってセグメンテーション結果を表示
    ax[1].set_title('Segmentation Result')
    ax[1].axis('off')  # 軸を非表示

    # 並べた画像を保存
    save_path = graph_save_dir / Path(image_path).stem  # 保存パスの設定
    plt.savefig(f'{save_path}_comparison.png', bbox_inches=None, pad_inches=0.1)  # 保存
    plt.close()  # メモリを節約するためにプロットを閉じる

def save_blended_image(save_dir, original_image_path, output_predictions, alpha=0.5):
    """
    元画像とセグメンテーション結果を重ね合わせ、指定された保存先に保存する関数。

    Args:
        output_predictions (np.ndarray): セグメンテーション結果の予測ラベル
        original_image_path (str): 元画像のパス
        save_dir (Path): 保存先のディレクトリ
        alpha (float): 元画像とセグメンテーション結果を重ねる割合（0.0～1.0）。デフォルトは0.5。
    """
    # 元画像の読み込み
    original_image = Image.open(original_image_path).convert('RGB')

    # output_predictions からカラー画像を作成
    output_colored = np.zeros((*output_predictions.shape, 3), dtype=np.uint8)
    for class_index, color in colors.items():
        if class_index == 0:  # 背景クラスを除く
            continue
        output_colored[output_predictions == class_index] = color

    # PIL画像に変換
    segmentation_image = Image.fromarray(output_colored)

    # セグメンテーション結果のサイズを元画像に合わせる
    segmentation_image_resized = segmentation_image.resize(original_image.size, resample=Image.NEAREST)

    # 元画像とセグメンテーション結果を重ね合わせ
    blended_image = Image.blend(original_image, segmentation_image_resized, alpha=alpha)

    # 保存パスの設定
    save_path = save_dir / f"{Path(original_image_path).stem}_blended.png"

    # 重ね合わせた画像を保存
    blended_image.save(save_path)

    print(f"Blended image saved to {save_path}")

def calculate_iou_from_confusion_matrix(conf_matrix, num_classes):
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
    num_classes: int = 5
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


# 混同行列を表示するためのCSV出力用関数
def save_confusion_matrix_with_metrics(conf_matrix, save_path, class_names):
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

# ランダムにデータセットから1つのデータを抽出して、グラフを作成する関数を定義します
def visualize_random_sample_from_dataset(dataset, save_path: Path):
    # ランダムに1つのインデックスを選択
    idx = random.randint(0, len(dataset) - 1)
    
    # データセットから画像とマスクを取得
    image, mask, image_name = dataset[idx]

    mean, std = dataset.get_mean_std()
    
    # 正規化を解除して画像を表示可能にする
    if mean is not None and std is not None:
        unnormalized_image = unnormalize(image, mean, std).permute(1, 2, 0).numpy()  # チャンネルを (H, W, C) に並べ替え
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)  # 0-1範囲から0-255範囲にスケーリング
    else:
        unnormalized_image = image
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = cv2.resize(mask, (unnormalized_image.shape[1], unnormalized_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # マスクのオーバーレイの処理 (3クラスに対応)
    mask_overlay = np.array(unnormalized_image)
    
    # マスクのクラスごとに異なる色を割り当てる
    mask_overlay[mask == 1, :] = [255, 0, 0]    # クラス1: 赤 (sellar)
    mask_overlay[mask == 2, :] = [0, 255, 0]    # クラス2: 緑 (sella)
    mask_overlay[mask == 3, :] = [255, 255, 0]  # クラス3: 黄 (pituitary)
    mask_overlay[mask == 4, :] = [128, 0, 128]  # クラス4: 紫 (tumor)
    mask_overlay[mask == 0, :] = [0, 0, 255]    # クラス0: 青 (背景)

    # 画像とマスクの表示
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(unnormalized_image)
    ax[0].set_title('Original Image')

    ax[1].imshow(mask_overlay)
    ax[1].set_title('Image with Mask Overlay')

    # グラフを保存
    save_path = Path(save_path)  # Path型で受け取る
    fig.savefig(save_path / f'visualized_sample_{image_name}.png', format='png')
    plt.close(fig)

# 正規化を元に戻すための関数
def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


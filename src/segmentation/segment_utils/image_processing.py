from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

colors = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (255, 255, 0),
    4: (128, 0, 128)
}

# クラス4のグラデーション用の色を定義
gradient_colors = [
    (230, 230, 250),  # 薄い紫（ラベンダー）
    (128, 0, 128),    # 普通の紫（元の基準色）
    (75, 0, 130)      # 濃い紫（インディゴ）
]

def segment_save(
    graph_save_dir: Path, 
    image_path: Path, 
    output_predictions: np.ndarray
) -> None:
    # 元画像の読み込み
    img = Image.open(image_path).convert('RGB')

    # ここでデータ型と形状を変換
    # uint8に変換して、0-255の範囲にスケールする
    if output_predictions.dtype != np.uint8:
        output_predictions = output_predictions.astype(np.uint8)

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

def save_blended_image(
    save_dir: Path, 
    original_image_path: Path, 
    output_predictions: np.ndarray, 
    alpha: float =0.5
) -> None:
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


def save_blended_image_with_class4_gradient(
    save_dir: Path, 
    original_image_path: Path, 
    output: torch.Tensor,
    alpha: float = 0.5
) -> None:
    """
    元画像とセグメンテーション結果を重ね合わせ、クラス4にのみ確率に基づいて3色のグラデーションをつけて保存する関数。

    Args:
        save_dir (Path): 保存先のディレクトリ
        original_image_path (Path): 元画像のパス
        output (torch.Tensor): モデルの出力（ロジット）
        alpha (float): 元画像とセグメンテーション結果を重ねる割合（0.0～1.0）。デフォルトは0.5。
    """
    # 元画像の読み込み
    original_image = Image.open(original_image_path).convert('RGB')

    # モデルの出力を元画像のサイズにリサイズ
    output_resized = F.interpolate(output, size=(original_image.height, original_image.width), mode='bilinear', align_corners=False)

    # リサイズした出力を確率に変換
    probabilities = F.softmax(output_resized, dim=1)
    
    # 確率をCPUに移動し、NumPy配列に変換
    probabilities = probabilities.cpu().numpy()

    # 閾値を定義（より高い値に設定）
    thresholds = [0.85, 0.90, 0.95]

    # 予測結果の画像を作成（元画像と同じサイズ）
    output_colored = np.zeros((original_image.height, original_image.width, 3), dtype=np.uint8)
    
    # 最も確率の高いクラスを選択
    pred_class = probabilities[0].argmax(axis=0)
    max_prob = probabilities[0].max(axis=0)

    for class_index in range(5):
        if class_index == 0:  # 背景クラスを除く
            continue
        if class_index == 4:
            # クラス4の場合、確率に基づいてグラデーションを適用
            mask = pred_class == class_index
            
            output_colored[mask & (max_prob < thresholds[0])] = gradient_colors[0]
            output_colored[mask & (max_prob >= thresholds[0]) & (max_prob < thresholds[1])] = gradient_colors[1]
            output_colored[mask & (max_prob >= thresholds[1])] = gradient_colors[2]
        else:
            # その他のクラスは固定色を使用
            output_colored[pred_class == class_index] = colors[class_index]

    # PIL画像に変換
    segmentation_image = Image.fromarray(output_colored)

    # 元画像とセグメンテーション結果を重ね合わせ
    blended_image = Image.blend(original_image, segmentation_image, alpha=alpha)

    # 保存パスの設定
    save_path = save_dir / f"{Path(original_image_path).stem}_blended_class4_gradient.png"

    # 重ね合わせた画像を保存
    blended_image.save(save_path)

    print(f"Blended image with class 4 gradient saved to {save_path}")



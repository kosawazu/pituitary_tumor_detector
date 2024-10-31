from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

colors = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (255, 255, 0),
    4: (128, 0, 128)
}

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
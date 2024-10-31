import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List

def visualize_random_sample_from_dataset(
    dataset: Dataset, 
    save_path: Path
) -> None:
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
def unnormalize(
    tensor: torch.Tensor, 
    mean: List[float], 
    std: List[float]
) -> torch.Tensor:
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def save_model_architecture(
    model: torch.nn.Module, 
    save_path: Path
) -> None:
    # モデルアーキテクチャを文字列化
    model_str = str(model)

    # ファイルに保存
    save_path = save_path / "model_architecture.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)  # ディレクトリが存在しない場合、作成する
    with open(save_path, 'w') as f:
        f.write(model_str)
    
    return
from pathlib import Path
import random

from pycocotools.coco import COCO
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

# fcn_resnet50用
transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file: Path, image_dir: Path, transform=None):
        """
        Args:
            annotation_file (Path): COCOフォーマットのアノテーションファイル（.json）
            image_dir (Path): 画像ファイルのディレクトリ
            transform (callable, optional): 画像に対して適用するトランスフォーム
        """
        self.coco = COCO(str(annotation_file))  # Pathオブジェクトを文字列に変換
        self.image_dir = image_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

        # もしリサイズトランスフォームが含まれていれば、リサイズの設定を抽出
        self.resize_transform = None
        if self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    self.resize_transform = t
                    break

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 画像IDを取得
        img_id = self.ids[idx]
        # COCOアノテーションから画像メタデータを取得
        img_metadata = self.coco.loadImgs(img_id)[0]
        # 画像のパスを取得 (Path型を使用)
        img_path = self.image_dir / img_metadata['file_name']
        
        # 画像の存在を確認、存在しない場合はスキップ
        if not img_path.exists():
            print(f"Image not found: {img_path}, skipping.")
            return None

        # 画像を読み込む
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Image not found: {img_path}, skipping.")
            return None

        # アノテーションIDを取得
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # アノテーションが存在しない場合はスキップ
        if len(anns) == 0:
            print(f"No annotations found for image: {img_path}, skipping.")
            return None

        # マスクを初期化（画像のサイズに対応した背景クラスのID 0 で初期化）
        mask = np.zeros((img_metadata['height'], img_metadata['width']), dtype=np.uint8)

        # 各アノテーションからセグメンテーションマスクを作成
        for ann in anns:
            m = self.coco.annToMask(ann)  # バイナリマスクを生成
            
            # カテゴリIDを取得
            category_id = ann['category_id']
            
            # カテゴリ名を取得して確認
            category_name = self.coco.loadCats(category_id)[0]['name']
            # print(f"Category ID: {category_id}, Category Name: {category_name}")
            
            # カテゴリIDに基づいてマスクを作成
            if category_id == 3:
                mask = np.maximum(mask, m * 3)  # カテゴリID 3 はクラス3に
            elif category_id == 4:
                mask = np.maximum(mask, m * 4)  # カテゴリID 4 はクラス4に

        # トランスフォームを適用（もし指定されていれば）
        if self.transform is not None:
            image = self.transform(image)  # 画像にリサイズ等のトランスフォームを適用

        # もしリサイズトランスフォームがあれば、マスクにも適用
        if self.resize_transform is not None:
            mask = Image.fromarray(mask)
            mask = self.resize_transform(mask)  # マスクを画像と同じサイズにリサイズ
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        else:
            mask = torch.as_tensor(mask, dtype=torch.int64)  # マスクをTensorに変換

        return image, mask, img_metadata['file_name']

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
        
        # カラーマッピングの設定（背景: 青, 紙袋: 緑, 傷: 赤）
        # ここで「紙袋」が緑で表示されるように設定
        colors = {
            0: (0, 0, 255),    # 背景 - 青
            1: (0, 255, 0),    # 紙袋 - 緑
            2: (255, 0, 0)     # 傷 - 赤
        }

        # カラーマップに基づいて output_predictions を色付け
        output_colored = np.zeros((*output_predictions.shape, 3), dtype=np.uint8)
        for class_index, color in colors.items():
            output_colored[output_predictions == class_index] = color

        # PIL画像に変換
        output_image = Image.fromarray(output_colored)

    # セグメンテーション結果のサイズを元画像に合わせる
    output_image_resized = output_image.resize(img.size, resample=Image.NEAREST)

    # プロットの設定（2つのサブプロットを横並び）
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

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
    
    # マスクのオーバーレイの処理 (3クラスに対応)
    mask_overlay = np.array(unnormalized_image)
    
    # マスクのクラスごとに異なる色を割り当てる
    mask_overlay[mask == 1, :] = [255, 0, 0]  # クラス1: 赤
    mask_overlay[mask == 2, :] = [0, 255, 0]  # クラス2: 緑
    mask_overlay[mask == 0, :] = [0, 0, 255]  # クラス0（背景）: 青

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


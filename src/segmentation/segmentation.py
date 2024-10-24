from pathlib import Path
import random

from pycocotools.coco import COCO
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt

# fcn_resnet50用
transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        colors = {
            0: (0, 0, 255),      # 背景 - 青
            1: (0, 255, 0),      # pituitary - 緑
            2: (255, 0, 0),      # sellar - 赤
            3: (255, 255, 0),    # tumor - 黄
            4: (128, 0, 128)     # sella - 紫
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
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = cv2.resize(mask, (unnormalized_image.shape[1], unnormalized_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # マスクのオーバーレイの処理 (3クラスに対応)
    mask_overlay = np.array(unnormalized_image)
    
    # マスクのクラスごとに異なる色を割り当てる
    mask_overlay[mask == 1, :] = [255, 0, 0]    # クラス1: 赤 (sellar)
    mask_overlay[mask == 2, :] = [0, 255, 0]    # クラス2: 緑 (pituitary)
    mask_overlay[mask == 3, :] = [255, 255, 0]  # クラス3: 黄 (tumor)
    mask_overlay[mask == 4, :] = [128, 0, 128]  # クラス4: 紫 (sella)
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


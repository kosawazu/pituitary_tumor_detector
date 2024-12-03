from pathlib import Path
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torch
import json
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from typing import Tuple

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_MAPPING = {
    0: "background",
    1: "sellar",
    2: "sella",
    3: "pituitary",
    4: "tumor",
}


class SegmentationDataset(Dataset):
    def __init__(
        self, 
        annotation_dir: Path, 
        image_dir: Path, 
        transform=None
    ):
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
    
class SubsetWithAttributes(Subset):
    def __getattr__(self, attr):
        return getattr(self.dataset, attr)
    
def save_filenames(dataset, indices, filepath):
    """
    指定されたインデックスに基づいてデータセット内のファイル名をテキストファイルに保存
    Args:
        dataset (Dataset): SegmentationDatasetオブジェクト
        indices (list): 保存するデータのインデックス
        filepath (Path): 保存先のファイルパス
    """
    # ファイル名を取得（annotation_filesから）
    filenames = [dataset.annotation_files[i].name for i in indices]
    with open(filepath, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")

def split_dataset(
    dataset: Dataset, 
    train_val_ratio: float =0.1, 
    test_size: int =10, 
    random_seed: int =42, 
    output_dir: Path =Path('./')
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    データセットを訓練、検証、テスト用に分割し、各データセットに含まれるファイル名をテキストファイルで保存する関数
    Args:
        dataset (Dataset): SegmentationDatasetオブジェクト
        train_val_ratio (float 0.~1.): 訓練から省く検証データの割合（テストデータを除いた部分からの割合）
        test_size (int): テストデータの枚数
        random_seed (int): ランダムシード
        output_dir (Path): 各データセットのファイル名を保存するディレクトリ
    
    Returns:
        train_dataset, val_dataset, test_dataset: 分割されたデータセット
    """
    assert isinstance(test_size, int), "test_sizeは整数で指定してください"

    # データセットのインデックスを取得
    indices = list(range(len(dataset)))

    # テストデータのインデックスを分割
    train_val_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_seed)

    # 残りのデータを訓練と検証に分割
    train_indices, val_indices = train_test_split(train_val_indices, test_size=train_val_ratio, random_state=random_seed)

    # ファイル名を保存するディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 各データセットに含まれるファイル名をテキストファイルに保存
    save_filenames(dataset, train_indices, output_dir / 'train_filenames.txt')
    save_filenames(dataset, val_indices, output_dir / 'val_filenames.txt')
    save_filenames(dataset, test_indices, output_dir / 'test_filenames.txt')

    # torch.utils.data.Subsetを使用してデータセットを分割
    train_dataset = SubsetWithAttributes(dataset, train_indices)
    val_dataset = SubsetWithAttributes(dataset, val_indices)
    test_dataset = SubsetWithAttributes(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

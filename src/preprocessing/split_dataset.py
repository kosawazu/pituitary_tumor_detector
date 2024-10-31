from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
from torch.utils.data import Dataset, Subset
from typing import Tuple

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

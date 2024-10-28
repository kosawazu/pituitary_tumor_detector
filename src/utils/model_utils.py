import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

def setup_fcn_model(
    model_name: str,
    num_classes: int =3,
    model_path: Path = None
) -> nn.Module:
    # モデルのロード（事前学習済みのモデルをファインチューニング）
    model_dict = {
    "fcn_resnet50":models.segmentation.fcn_resnet50(pretrained=True),
    "fcn_resnet101":models.segmentation.fcn_resnet101(pretrained=True),
    "fcn_vgg16": models.vgg16(pretrained=True),
    "fcn_vgg19": models.vgg19(pretrained=True)
    }
    if model_name not in model_dict:
        raise ValueError(f"Invalid model_name '{model_name}'.")
    model = model_dict[model_name]

    # クラス数を紙袋検出用に調整（背景+紙袋 = 2クラス）
    if "resnet" in model_name:
        # ResNetの場合（512チャンネル）
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif "vgg" in model_name:
        # VGGの特徴抽出部を取得
        features = list(model.features.children())
        
        # VGGの全結合層部分を削除し、畳み込み層に置き換え
        classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),  # FCN部分の追加
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, num_classes, kernel_size=1)  # 出力層
        )

        # VGGモデルをセグメンテーションに適した形で再構築
        model = nn.Sequential(
            *features,
            classifier
        )

        # 出力を辞書形式で返すように調整
        class VGG_FCN(nn.Module):
            def __init__(self, vgg_model):
                super(VGG_FCN, self).__init__()
                self.vgg_model = vgg_model

            def forward(self, x):
                # VGGモデルの出力をセグメンテーション形式に整える
                x = self.vgg_model(x)
                return {"out": x}  # 辞書形式で出力

        # ラップして辞書形式で返すモデルに変換
        model = VGG_FCN(model)

    return model

def setup_detection_model(
    model_name: str,
    num_classes: int =3
) -> nn.Module:
    # モデルのロード（事前学習済みのモデルをファインチューニング）
    model_dict = {
        "ssd300_vgg16": models.detection.ssd300_vgg16(pretrained=True),
        # 必要に応じて他のSSDモデルも追加可能
    }
    if model_name not in model_dict:
        raise ValueError(f"Invalid model_name '{model_name}'.")
    model = model_dict[model_name]

    # クラス数を紙袋検出用に調整（背景+紙袋 = 2クラス）
    in_features = model.head.classification_head.conv[0].in_channels
    model.head.classification_head.num_classes = num_classes
    model.head.classification_head.conv = nn.Conv2d(
        in_features, num_classes * 4, kernel_size=3, padding=1
    )  # SSDの出力層をクラス数に応じて変更
    return model

def setup_device(
    model: nn.Module,
    model_path: Path = None
) -> torch.nn.Module:
    """モデルをデバイス（GPU/CPU）に設定し、必要に応じてマルチGPUモードに切り替えます。"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            logger.info(f"Let's use {device_count} GPUs!")
            model = torch.nn.DataParallel(model)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.info("CUDA is not available. Using CPU.")
    model.to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return device, model

#EarlyStopping
class EarlyStopping:
    def __init__(
        self, 
        patience: int=10, 
        verbose: int=0
        ) -> None:
        '''
        Parameters:
            patience(int): 監視するエポック数(デフォルトは10)
            verbose(int): 早期終了の出力フラグ
                          出力(1),出力しない(0)        
        '''

        self.count = 0 # 監視中のエポック数のカウンターを初期
        self.pre_loss = float('inf') # 比較対象の損失を無限大'inf'で初期化
        self.patience = patience # 監視対象のエポック数をパラメーターで初期化
        self.verbose = verbose # 早期終了メッセージの出力フラグをパラメーターで初期化
        
    def __call__(
        self, 
        current_loss: float
        ) -> bool:
        '''
        Parameters:
            current_loss(float): 1エポック終了後の検証データの損失
        Return:
            True:監視回数の上限までに前エポックの損失を超えた場合
            False:監視回数の上限までに前エポックの損失を超えない場合
        '''
        
        if self.pre_loss < current_loss: # 前エポックの損失より大きくなった場合
            self.count += 1 # カウンターを1増やす

            if self.count > self.patience: # 監視回数の上限に達した場合
                if self.verbose:  # 早期終了のフラグが1の場合
                    logger.info('early stopping')
                return True # 学習を終了するTrueを返す
            
        else: # 前エポックの損失以下の場合
            self.count = 0 # カウンターを0に戻す
            self.pre_loss = current_loss # 損失の値を更新する
        
        return False
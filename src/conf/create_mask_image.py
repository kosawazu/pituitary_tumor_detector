from PIL import Image, ImageDraw
import numpy as np
import json
from pathlib import Path

def process_segmentation_from_json(annotation_dir: Path, image_dir: Path, output_dir: Path, alpha=0.5):
    # JSONファイルが含まれるディレクトリ内のすべてのJSONファイルを取得
    annotation_files = list(Path(annotation_dir).glob('*.json'))

    # 出力ディレクトリを作成（存在しない場合は作成）
    output_dir.mkdir(parents=True, exist_ok=True)

    # 優先順位の高い順にタグを並べる
    priority_tags = ["sellar", "sella", "pituitary", "tumor"]

    # 各クラスに対応する色を設定
    color_map = {
        "pituitary": (255, 255, 0),  # pituitary - 黄
        "sellar": (0, 255, 0),     # sellar - 緑
        "tumor": (128, 0, 128),      # tumor - 紫
        "sella": (255, 0, 0),    # sella - 赤
    }

    for annotation_file in annotation_files:
        # JSONファイルを読み込み
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # 画像ファイル名を取得
        img_metadata = data['asset']
        img_filename = img_metadata['name']  # ファイル名のみ取得

        # 画像のフルパスを作成 (image_dir からファイル名を探す)
        img_path = image_dir / img_filename

        # 画像を読み込む
        try:
            original_image = Image.open(img_path).convert('RGB')  # 画像はPIL形式で読み込む
        except FileNotFoundError:
            print(f"Image not found: {img_path}, skipping.")
            continue

        # 元画像のサイズに基づいて空のカラーマスクを作成
        mask = Image.new('RGB', (img_metadata['size']['width'], img_metadata['size']['height']))

        # アノテーション情報から領域を取得し、指定の優先度順に描画
        regions = data.get('regions', [])
        for tag in priority_tags:
            for region in regions:
                if tag in region['tags']:
                    points = region['points']
                    polygon = [(point['x'], point['y']) for point in points]

                    # 領域を塗りつぶす
                    img_mask = Image.new('L', (img_metadata['size']['width'], img_metadata['size']['height']), 0)
                    ImageDraw.Draw(img_mask).polygon(polygon, outline=1, fill=1)
                    region_mask = np.array(img_mask)

                    # 領域のタグに基づいてクラスごとの色を適用
                    colored_region = Image.new('RGB', (img_metadata['size']['width'], img_metadata['size']['height']), color=color_map[tag])
                    colored_region_np = np.array(colored_region)

                    # マスクに塗りつぶし
                    mask = np.where(region_mask[:, :, None], colored_region_np, mask)

        # NumPy配列からPIL画像に変換
        mask_image = Image.fromarray(mask.astype(np.uint8))

        # 元画像とマスクをブレンド
        blended_image = Image.blend(original_image, mask_image, alpha=alpha)

        # 出力ファイルパスを作成
        output_filename = f"segmented_{img_filename}"
        output_image_path = output_dir / output_filename

        # 結果を保存
        blended_image.save(output_image_path)
        print(f"Saved segmented image: {output_image_path}")


# 例: 使用方法
annotation_dir = Path('../../data/mask_json')  # JSONファイルがあるディレクトリ
image_dir = Path('../../data/img')            # 画像ファイルがあるディレクトリ
output_dir = Path('../../data/mask_image_py')           # 出力先のディレクトリ

process_segmentation_from_json(annotation_dir, image_dir, output_dir)

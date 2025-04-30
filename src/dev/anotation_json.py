import os
import json
from collections import defaultdict
from pathlib import Path

def extract_class_ids_from_json(json_folder):
    class_id_map = defaultdict(set)  # クラス名とIDをセットで保存

    # JSONフォルダ内のすべてのファイルを処理
    for json_file in Path(json_folder).glob('*.json'):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file}: {e}")
                continue  # 読み込みエラーが発生した場合スキップ

            # アノテーション領域を確認し、クラスとIDを取得
            regions = data.get('regions', [])
            if not regions:
                print(f"No regions found in {json_file}")
            
            for region in regions:
                tags = region.get('tags', [])
                region_id = region.get('id', None)
                
                # デバッグ: タグとIDを表示
                if tags and region_id:
                    print(f"Tags: {tags}, Region ID: {region_id}")

                for tag in tags:
                    class_id_map[tag].add(region_id)

    return class_id_map

# フォルダのパスを指定
json_folder = '../../data/mask_json'  # mask_jsonフォルダへのパス
class_id_map = extract_class_ids_from_json(json_folder)

# 結果を表示
if class_id_map:
    for class_name, ids in class_id_map.items():
        print(f"Class: {class_name}, IDs: {', '.join(ids)}")
else:
    print("No classes or IDs found.")

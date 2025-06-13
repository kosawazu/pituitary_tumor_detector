#!/bin/bash

echo "Script started in directory: $(pwd)"
echo "Script needed to start in 'pituitary_tumor_detector/src/segmentation'"

model_name="deeplabv3_resnet101"
model_path="../../best_model/$model_name/best_tumor_iou_model.pth"
image_dir="../../data/kenshou_kekka/kenshou2_takeuchi/2_1fps"
json_path="../../data/kenshou_kekka/kenshou2_takeuchi/vott-json-export/kenshou2_takeuchi-export.json"
save_dir="../../result/ext_val"

# テストの実行とログ出力
python3 evaluate_iou_sequence.py --model_name "$model_name" --model_path $model_path --image_dir $image_dir --json_path $json_path --save_dir $save_dir 

echo "全ての推論と学習が終了しました"

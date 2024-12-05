#!/bin/bash

if [ $# -ne 6 ]; then
    echo "Error: Exactly 6 arguments are required."
    echo "Usage: $0 <model_name> <epochs> <batch_size_list> <learning_rate_list> <patience> <attention_mode>"
    exit 1
fi

model_name="$1"
epochs="$2"
batch_size_list=($3)  # リストとして展開
learning_rate_list=($4)  # リストとして展開 
patience="$5"
attention_mode="$6"

model_files=("best_loss_model" "best_tumor_iou_model" "best_mean_iou_model")
# model_files=("best_loss_model")
# ログフォルダのパスを定義
train_log_dir="../../logs/train"
test_log_dir="../../logs/test"

# ログフォルダが存在しない場合、作成
mkdir -p "$train_log_dir"
mkdir -p "$test_log_dir"

# バッチサイズと学習率のリストをループで回す
for batch_size in "${batch_size_list[@]}"; do
  for learning_rate in "${learning_rate_list[@]}"; do
    echo "バッチサイズ: $batch_size, 学習率: $learning_rate"

    # フォルダのパスを定義
    result_folder="../../result/nagoya/training_results/${model_name}/${batch_size}/${learning_rate}"
    echo "結果保存先のディレクトリ： $result_folder"
    # フォルダが存在する場合はスキップ
    if [ -d "$result_folder" ]; then
      echo "フォルダ $result_folder が既に存在します。次の処理に進みます。"
      continue
    fi

    # ログファイル名を定義
    log_file_name="${model_name}_${batch_size}_${learning_rate}.log"
    # トレーニングの実行とログ出力
    python3 train_fcn.py --epochs "$epochs" --model_name "$model_name" --batch_size "$batch_size" --learning_rate "$learning_rate" --patience "$patience" --attention_mode $attention_mode > "${train_log_dir}/${log_file_name}" 2>&1
    
    # トレーニングが正常に終了したかをチェック
    if [ $? -ne 0 ]; then
      echo "トレーニング中にエラーが発生しました。次のバッチサイズと学習率の組み合わせに進みます。"
      continue  # エラーが発生した場合、次のループに進む
    fi

    for model_file in "${model_files[@]}"; do
      # テストの実行とログ出力
      python3 test_fcn.py --model_name "$model_name" --save_model_file "$model_file" --batch_size "$batch_size" --learning_rate "$learning_rate" --attention_mode $attention_mode > "${test_log_dir}/${log_file_name}" 2>&1
    done
  done
done

echo "全ての推論と学習が終了しました"

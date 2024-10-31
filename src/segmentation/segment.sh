model_name="$1"
epochs="$2"
batch_size_list=($3)  # リストとして展開
learning_rate_list=($4)  # リストとして展開 
patience="$5"

# バッチサイズと学習率のリストをループで回す
for batch_size in "${batch_size_list[@]}"; do
  for learning_rate in "${learning_rate_list[@]}"; do
    echo "バッチサイズ: $batch_size, 学習率: $learning_rate"

    # フォルダのパスを定義
    result_folder="../../result/nagoya/training_results/${model_name}/${batch_size}/${learning_rate}"

    # フォルダが存在する場合はスキップ
    if [ -d "$result_folder" ]; then
      echo "フォルダ $result_folder が既に存在します。次の処理に進みます。"
      continue
    fi

    # トレーニングの実行
    python3 train_fcn.py --epochs "$epochs" --model_name "$model_name" --batch_size "$batch_size" --learning_rate "$learning_rate" --patience "$patience"

    # テストの実行
    python3 test_fcn.py --model_name "$model_name" --batch_size "$batch_size" --learning_rate "$learning_rate"
  done
done

echo "全ての推論と学習が終了しました"

model_name="$1"
epochs="$2"
batch_size_list=($3)  # リストとして展開
learning_rate_list=($4)  # リストとして展開 
patience="$5"

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

    # フォルダが存在する場合はスキップ
    if [ -d "$result_folder" ]; then
      echo "フォルダ $result_folder が既に存在します。次の処理に進みます。"
      continue
    fi

    # ログファイル名を定義
    log_file_name="${model_name}_${batch_size}_${learning_rate}.log"

    # トレーニングの実行とログ出力
    python3 train_fcn.py --epochs "$epochs" --model_name "$model_name" --batch_size "$batch_size" --learning_rate "$learning_rate" --patience "$patience" > "${train_log_dir}/${log_file_name}" 2>&1
    
    # トレーニングが正常に終了したかをチェック
    if [ $? -ne 0 ]; then
      echo "トレーニング中にエラーが発生しました。次のバッチサイズと学習率の組み合わせに進みます。"
      continue  # エラーが発生した場合、次のループに進む
    fi

    # テストの実行とログ出力
    python3 test_fcn.py --model_name "$model_name" --batch_size "$batch_size" --learning_rate "$learning_rate" > "${test_log_dir}/${log_file_name}" 2>&1
  done
done

echo "全ての推論と学習が終了しました"

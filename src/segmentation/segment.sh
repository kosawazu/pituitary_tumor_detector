# コマンドライン引数を変数に格納
model_name="$1"
epochs="$2"
batch_size="$3"
learning_rate="$4"
class_num="$5"

# トレーニングの実行
python3 train_fcn.py --epochs "$epochs" --model_name "$model_name" --batch_size "$batch_size" --learning_rate "$learning_rate" --class_num "$class_num"

# テストの実行
python3 test_fcn.py --model_name "$model_name" --class_num "$class_num" --batch_size "$batch_size" --learning_rate "$learning_rate"
model_name="fcn_resnet101"
epochs=50
batch_size=64
lerning_rate=1e-4
class_num=3

python3 train_fcn.py --epochs "$epochs" --model_name "$model_name" --batch_size "$batch_size" --lerning_rate "$lerning_rate" --class_num "$class_num"

python3 test_fcn.py --model_name "$model_name" --class_num "$class_num" --batch_size "$batch_size" --lerning_rate "$lerning_rate"
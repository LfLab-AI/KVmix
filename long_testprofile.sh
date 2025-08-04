gpuid=0
group_size=32
model='/path/your_model'

CUDA_VISIBLE_DEVICES=$gpuid python pred_long_benchprofile.py  --model_name_or_path $model \
    --cache_dir ./cached_models \
    --group_size $group_size \

gpuid=0
group_size=32
model='/home/dell/lf/lf/model/meta-llama/Llama-2-7b-hf' ##'/home/dell/lf/lf/model/Mistral-7B-Instruct-v0.3' ##'/home/dell/lf/lf/model/meta-llama/Llama-2-7b-hf' ##'/home/dell/lf/lf/model/Mistral-7B-Instruct-v0.3' ##'/home/dell/lf/lf/model/Meta-Llama-3-8B'

CUDA_VISIBLE_DEVICES=$gpuid python pred_long_benchprofile.py  --model_name_or_path $model \
    --cache_dir ./cached_models \
    --group_size $group_size \

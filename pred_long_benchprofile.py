import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
os.environ["WANDB_DISABLED"] = "true"

from utils.process_args import process_args
from transformers import LlamaConfig, MistralConfig, AutoTokenizer

from myprofileKV import profile_model
import gc

def build_chat(tokenizer, prompt, model_name):
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    model2maxlen = json.load(open("configs/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_args, data_args, training_args = process_args()
    model_name = model_args.model_name_or_path.split("/")[-1]
    dtype = torch.float16
    
    # load tokenizer
    if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                            trust_remote_code=True)
        # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
        #                                     use_fast=False, 
        #                                     trust_remote_code=True, 
        #                                     tokenizer_type='llama')
    elif 'mistral' in model_args.model_name_or_path.lower():
        config = MistralConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True)
    else:
        raise NotImplementedError
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    datasets = ["triviaqa", "qasper", "trec", "lcc", "repobench-p", "qmsum", "2wikimqa", "passage_retrieval_en", "gov_report", "multifieldqa_en"]
    
    dataset2maxlen = json.load(open("configs/dataset2maxlen.json", "r"))
    dataset2prompt = json.load(open("configs/dataset2prompt.json", "r"))
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        data = load_dataset('path/dataset/LongBench/', dataset, split='test')
        
        # Randomly select number prompt from the data set
        number_random = 10
        prompts = []
        num_samples = min(number_random, len(data)) 
        random_indices = random.sample(range(len(data)), num_samples)
        for i in random_indices:
            json_obj = data[i]
            prompt_format = dataset2prompt[dataset]
            prompt = prompt_format.format(**json_obj)
            prompts.append(prompt)
        
        # 准备输入数据
        inputs = tokenizer(prompts, add_special_tokens=False, return_tensors='pt', padding=True)
        input_ids = inputs.input_ids  
        attention_mask = inputs.attention_mask 
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # Shift left one position
        labels[:, -1] = tokenizer.pad_token_id  # The last position is pad_token
        
        # Convert the input into a list, with each prompt being a separate tensor
        input_ids_list = [input_ids[i].unsqueeze(0).to(device) for i in range(number_random)]
        attention_mask_list = [attention_mask[i].unsqueeze(0).to(device) for i in range(number_random)]
        labels_list = [labels[i].unsqueeze(0).to(device) for i in range(number_random)]
        
        # load full precision model for profiling
        if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="cuda:0",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
        elif 'mistral' in model_args.model_name_or_path.lower():
            from transformers import MistralForCausalLM
            model = MistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="cuda:0",
            )
        else:
            raise NotImplementedError
        
        # start profile
        num_layers = config.num_hidden_layers
        quant_bits = profile_model(model, input_ids_list, attention_mask_list, labels_list, num_layers)
        
        # delete model and related tensors
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # load kv-mix model
        if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
            from models.modeling_llama_KVmix import LlamaForCausalLM_KVmix
            config.group_size = model_args.group_size
            config.use_flash = False
            model = LlamaForCausalLM_KVmix.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="cuda:0",
                quant_bits=quant_bits,  # quant_bits obtained using profiler
            )
        elif 'mistral' in model_args.model_name_or_path.lower():
            from models.modeling_mistral_KVmix import MistralForCausalLM_KVmix
            config.group_size = model_args.group_size
            config.use_flash = False
            model = MistralForCausalLM_KVmix.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                attn_implementation="eager",
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="cuda:0",
                quant_bits=quant_bits,  # quant_bits obtained using profiler
            )
        else:
            raise NotImplementedError
        
        model.eval()
        max_length = model2maxlen[model_name]
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        
        out_dir = f"pred/{model_name}"
        out_path = f"{out_dir}/{dataset}.jsonl"
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

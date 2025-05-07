import argparse
import json
import numpy as np
import torch
from transformers import AutoTokenizer, TextStreamer, LlamaConfig, LlamaForCausalLM
from models.modeling_llama_KVmix import LlamaForCausalLM_KVmix
from models.modeling_mistral_KVmix import MistralForCausalLM_KVmix
from myprofileKV import profile_model
import gc
from models.modeling_llama import LlamaForCausalLM

GROUP_SIZE = 32
PATH_TO_YOUR_SAVE_DIR = './cached_models'
config = LlamaConfig.from_pretrained('/home/dell/lf/lf/model/meta-llama/Llama-2-7b-hf')
config.group_size = GROUP_SIZE
config.use_flash = False
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='/home/dell/lf/lf/model/meta-llama/Llama-2-7b-hf')
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--KVmix_bsz", type=int, default=28)
    parser.add_argument("--original_bsz", type=int, default=4)
    parser.add_argument("--KVmix_enable", default=True, action="store_true")
    parser.add_argument("--original_enable", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    set_seed(args)

    prompt_text1 = """There is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: <|im_end|>\n<|im_start|>assistant\nB\n<|im_end|>\n<|im_start|>user\nThere is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: <|im_end|>\n<|im_start|>assistant\nB\n<|im_end|>\n<|im_start|>user\nThere is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: <|im_end|>\n<|im_start|>assistant\nC\n<|im_end|>\n<|im_start|>user\nThere is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: <|im_end|>\n<|im_start|>assistant\nA\n<|im_end|>\n<|im_start|>user\nThere alentours a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Find the characteristic of the ring 2Z.\nA. 0\nB. 3\nC. 12\nD. 30\nAnswer: <|im_end|>\n<|im_start|>assistant\nA\n<|im_end|>\n<|im_start|>user\nThere is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer: """
    prompt_text = prompt_text1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    prompt_encoded = tokenizer(prompt_text, add_special_tokens=False)
    prompt_token_length = len(prompt_encoded.input_ids)
    print(f"Prompt Token Length: {prompt_token_length}")

    if args.KVmix_enable:
        inference_inputs = tokenizer([prompt_text] * 1, add_special_tokens=False, return_tensors='pt', padding=True)
        inference_input_ids = inference_inputs.input_ids
        inference_attention_mask = inference_inputs.attention_mask
        # Create labels for the generative task (autoregressive: labels are input_ids shifted left)
        labels = inference_input_ids.clone()
        labels[:, :-1] = inference_input_ids[:, 1:]
        labels[:, -1] = tokenizer.pad_token_id

        KVmix_model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
        )
        KVmix_model.eval()

        input_ids_list = [inference_input_ids[i].unsqueeze(0).to(KVmix_model.device) for i in range(1)]
        attention_mask_list = [inference_attention_mask[i].unsqueeze(0).to(KVmix_model.device) for i in range(1)]
        labels_list = [labels[i].unsqueeze(0).to(KVmix_model.device) for i in range(1)]

        num_layers = config.num_hidden_layers
        quant_bits = profile_model(
            KVmix_model,
            input_ids_list,
            attention_mask_list,
            labels_list,
            num_layers
        )

        print("Profiling completed, quantization bits set.")
        print("Memory before cleanup (MB): ", f"{torch.cuda.memory_allocated(device=KVmix_model.device) / 1024 / 1024:.3f}")

        del KVmix_model, inference_input_ids, inference_attention_mask, inference_inputs, labels
        torch.cuda.empty_cache()
        gc.collect()

        print("Memory after cleanup (MB): ", f"{torch.cuda.memory_allocated(device='cuda:0') / 1024 / 1024:.3f}")
        print("quant_bits=", quant_bits)

        KVmix_model = LlamaForCausalLM_KVmix.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
            attn_implementation="eager",
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
            quant_bits=quant_bits,
        )
        KVmix_model.eval()

        inference_inputs = tokenizer([prompt_text] * args.KVmix_bsz, add_special_tokens=False, return_tensors='pt', padding=True)
        inference_input_ids = inference_inputs.input_ids
        inference_attention_mask = inference_inputs.attention_mask

        print("My quant Model GPU Memory Per GPU (MB): ", f"{torch.cuda.max_memory_allocated(device=KVmix_model.device) / 1024 / 1024:.3f}")
        KVmix_model.config.pad_token_id = tokenizer.pad_token_id

        single_input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(KVmix_model.device)
        print("\n")
        print("############# GPU Warm Up ... #############")
        print("\n")
        for i in range(5):
            generate_ids = KVmix_model.generate(single_input_ids, max_new_tokens=16)

        del single_input_ids, generate_ids
        torch.cuda.empty_cache()
        gc.collect()

        print("\n")
        print("############# Running Myquant Model #############")
        print("Batch Size: ", args.KVmix_bsz)
        print("Input Tokens Num: ", inference_input_ids.shape[1])
        print("Max New Tokens: ", args.max_new_tokens)
        
        streamer = TextStreamer(tokenizer, skip_prompt=False) if args.KVmix_bsz == 1 else None
        torch.cuda.reset_peak_memory_stats(device=KVmix_model.device)
        memory_before = torch.cuda.memory_allocated(device=KVmix_model.device)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        generate_ids = KVmix_model.generate(
            inference_input_ids.to(KVmix_model.device),
            attention_mask=inference_attention_mask.to(KVmix_model.device),
            streamer=streamer,
            max_new_tokens=args.max_new_tokens
        )
        end_time.record()
        torch.cuda.synchronize()
        KVmix_wall_time = start_time.elapsed_time(end_time)
        KVmix_max_gpu_memory = torch.cuda.max_memory_allocated(device=KVmix_model.device)

        memory_after = torch.cuda.max_memory_allocated(device=KVmix_model.device)
        kv_cache_memory = memory_after - memory_before
        print(f"Estimated KV Cache Memory: {kv_cache_memory / 1024 / 1024:.3f} MB")

        all_token_num = generate_ids.shape[1] * generate_ids.shape[0]
        print("\n--------------------------------")
        print(f"Wall Time (ms): {KVmix_wall_time:.3f}")
        print(f"Total Token Num: {all_token_num}")
        print(f"Througthput (token/s): {all_token_num / KVmix_wall_time * 1000:.3f}")
        print(f"Latency (ms/token): {KVmix_wall_time / all_token_num:.3f}")
        print(f"Max GPU Memory Per GPU (MB): {KVmix_max_gpu_memory / 1024 / 1024:.3f}")

        del generate_ids
        torch.cuda.empty_cache()
        gc.collect()
    # Loading the original Llama model for efficiency and memory comparison
    if args.original_enable:
        original_model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
            attn_implementation="eager",
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
        )
        original_model.eval()

        inference_inputs = tokenizer([prompt_text] * args.original_bsz, add_special_tokens=False, return_tensors='pt', padding=True)
        inference_input_ids = inference_inputs.input_ids
        inference_attention_mask = inference_inputs.attention_mask

        print("Original Model GPU Memory Per GPU (MB): ", f"{torch.cuda.max_memory_allocated(device=original_model.device) / 1024 / 1024:.3f}")
        original_model.config.pad_token_id = tokenizer.pad_token_id

        single_input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(original_model.device)
        print("\n")
        print("############# GPU Warm Up ... #############")
        print("\n")
        for i in range(5):
            generate_ids = original_model.generate(single_input_ids, max_new_tokens=16)

        del single_input_ids, generate_ids
        torch.cuda.empty_cache()
        gc.collect()

        print("\n")
        print("############# Running Original Model #############")
        print("Batch Size: ", args.original_bsz)
        print("Input Tokens Num: ", inference_input_ids.shape[1])
        print("Max New Tokens: ", args.max_new_tokens)
        
        streamer = TextStreamer(tokenizer, skip_prompt=False) if args.original_bsz == 1 else None
        torch.cuda.reset_peak_memory_stats(device=original_model.device)
        memory_before = torch.cuda.memory_allocated(device=original_model.device)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        generate_ids = original_model.generate(
            inference_input_ids.to(original_model.device),
            attention_mask=inference_attention_mask.to(original_model.device),
            streamer=streamer,
            max_new_tokens=args.max_new_tokens
        )
        end_time.record()
        torch.cuda.synchronize()
        original_wall_time = start_time.elapsed_time(end_time)
        original_max_gpu_memory = torch.cuda.max_memory_allocated(device=original_model.device)

        memory_after = torch.cuda.max_memory_allocated(device=original_model.device)
        kv_cache_memory = memory_after - memory_before
        print(f"Estimated KV Cache Memory: {kv_cache_memory / 1024 / 1024:.3f} MB")

        all_token_num = generate_ids.shape[1] * generate_ids.shape[0]
        print("\n--------------------------------")
        print(f"Wall Time (ms): {original_wall_time:.3f}")
        print(f"Total Token Num: {all_token_num}")
        print(f"Througthput (token/s): {all_token_num / original_wall_time * 1000:.3f}")
        print(f"Latency (ms/token): {original_wall_time / all_token_num:.3f}")
        print(f"Max GPU Memory Per GPU (MB): {original_max_gpu_memory / 1024 / 1024:.3f}")

    else:
        print("Please specify either --KVmix_enable or --original_enable")
        return

if __name__ == "__main__":
    main()
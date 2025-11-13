# KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache


## Overview

This repository contains the implementation of KVmix, a novel gradient-guided mixed-precision quantization method for optimizing the Key-Value (KV) Cache in Large Language Models (LLMs). KVmix leverages gradient-based importance analysis to dynamically allocate quantization bits, ensuring higher precision for critical layers while aggressively compressing less significant ones. It also integrates dynamic critical context optimization and supports users to perform 1, 2, 3, and 4-bit mixed quantization.

<img src = "KVmixfig/KVmix profiler.png" align = "center" width="100%" hight="100%">

# - [ ] **Our work has been accepted by AAAI 2026 (oral!).**

## Getting Started

### Installation
1. Create a conda environment
```
conda create --name kvmix python=3.10 -y
conda activate kvmix
```
2. Clone and install the dependencies
```
cd KVmix
pip install -r requirements.txt
```
3. Installing CUDA Implementation
```
cd quant_cuda
pip install -e .
```
### Use KVmix Profiler
myprofileKV.py is our KVmix profiler implementation. You can easily use it and modify the configuration in myprofileKV.py according to your needs.
Examples of use：
```
from myprofileKV import profile_model
from models.modeling_llama_KVmix import LlamaForCausalLM_KVmix
inference_inputs = tokenizer([prompt_text], add_special_tokens=False, return_tensors='pt', padding=True)
inference_input_ids = inference_inputs.input_ids
inference_attention_mask = inference_inputs.attention_mask

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

input_ids_list = [inference_input_ids[i].unsqueeze(0).to(KVmix_model.device) for i in range(n)]
attention_mask_list = [inference_attention_mask[i].unsqueeze(0).to(KVmix_model.device) for i in range(n)]
labels_list = [labels[i].unsqueeze(0).to(KVmix_model.device) for i in range(n)]

num_layers = config.num_hidden_layers
quant_bits = profile_model(
            KVmix_model,
            input_ids_list,
            attention_mask_list,
            labels_list,
            num_layers
        )
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
```
### Inference memory and efficiency test
Use Inference_comparision.py to test the memory reduction and efficiency improvement of KVmix compared to the FP16 model. You can test the maximum throughput of the model by changing the batch size.
```
python Inference-comparision.py --model_name_or_path /path/Llama-2-7b-hf
```
### Evaluate KVmix on LongBench
We support using llama and mistral models to evaluate the performance of KVmix on LongBench. You can modify the number of 3/4bit quantization layers in the KVmix profiler according to your requirements for memory compression ratio and model accuracy.
```
sh scripts/Longbenchtest_profiler.sh
python eval_longbench.py --model {model}
```

### Evaluate KVmix by lm_eval
We support using [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate KVmix performance, but you need to replace the huggingface model in lm-evaluation-harness with a model quantized using KVmix.
```
cd lm-evaluation-harness
pip install -e .
sh eval.sh
```




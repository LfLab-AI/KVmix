import torch
import numpy as np
import gc
import matplotlib.pyplot as plt
import os
import seaborn as sns

import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.pyplot as plt


def profile_importance(model, input_ids_list, attention_mask_list, labels_list, num_layers, max_length=1024):
    """Calculate the importance scores of each layer's Key and Value based on the gradient and handle multiple prompts"""
    all_k_importance_scores = []
    all_v_importance_scores = []
    
    for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
        # Limit input length to avoid OOM
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
            labels = labels[:, :max_length]

        model.train()
        k_importance_scores = []
        v_importance_scores = []
        
        for layer in range(num_layers):
            model.zero_grad()  
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Calculate the gradient for the key
            k_params = [model.model.layers[layer].self_attn.k_proj.weight]
            torch.autograd.backward(loss, create_graph=False, retain_graph=True, inputs=k_params)
            k_grad_norm = sum(p.grad.norm(2).item() for p in k_params if p.grad is not None)
            k_importance_scores.append(k_grad_norm)
            
            # Calculate the gradient for the value
            v_params = [model.model.layers[layer].self_attn.v_proj.weight]
            torch.autograd.backward(loss, create_graph=False, retain_graph=True, inputs=v_params)
            v_grad_norm = sum(p.grad.norm(2).item() for p in v_params if p.grad is not None)
            v_importance_scores.append(v_grad_norm)
            
            # Clean up memory
            del outputs, loss
            torch.cuda.empty_cache()
            gc.collect()
        
        all_k_importance_scores.append(k_importance_scores)
        all_v_importance_scores.append(v_importance_scores)

    # Calculate the average importance score of all prompts
    k_importance_scores_mean = np.mean(all_k_importance_scores, axis=0)
    v_importance_scores_mean = np.mean(all_v_importance_scores, axis=0)
    return torch.tensor(k_importance_scores_mean), torch.tensor(v_importance_scores_mean)

def classify_layers(importance_scores, num_layers):
    """Classify the layers into important (10%), general (20%) and unimportant (70%)"""
    sorted_indices = torch.argsort(importance_scores, descending=True)
    num_important = int(0.1 * num_layers)
    num_general = int(0.1 * num_layers)
    important_layers = sorted_indices[:num_important].tolist()
    general_layers = sorted_indices[num_important:num_important + num_general].tolist()
    unimportant_layers = sorted_indices[num_important + num_general:].tolist()
    return important_layers, general_layers, unimportant_layers


def set_quant_bits(k_important, k_general, k_unimportant, v_important, v_general, v_unimportant):
    """Set the quantization bit and full precision ratio according to the importance of Key and Value"""
    quant_bits = {}
    total_layers = len(k_important) + len(k_general) + len(k_unimportant)
    
    residual_ratios = {
        4: 0.3,  
        2: 0.2, 
        1: 0.1,
        3: 0.2
    }
    #The quantization bit and full precision ratio can be adjusted according to needs.
    for layer in range(total_layers):
        if layer in k_important:
            k_bits = 3 
        elif layer in k_general:
            k_bits = 3  
        else:
            k_bits = 2 
        k_residual_ratio = residual_ratios[k_bits]

        if layer in v_important:
            v_bits = 4  
        elif layer in v_general:
            v_bits = 4 
        else:
            v_bits = 2 
        v_residual_ratio = residual_ratios[v_bits]

        quant_bits[layer] = {
            'k_bits': k_bits,
            'v_bits': v_bits,
            'k_residual_ratio': k_residual_ratio,
            'v_residual_ratio': v_residual_ratio
        }
    return quant_bits


def profile_model(model, input_ids_list, attention_mask_list, labels_list, num_layers):
    k_importance_scores, v_importance_scores = profile_importance(
        model, input_ids_list, attention_mask_list, labels_list, num_layers, max_length=1024
    )
    k_important, k_general, k_unimportant = classify_layers(k_importance_scores, num_layers)
    v_important, v_general, v_unimportant = classify_layers(v_importance_scores, num_layers)
    quant_bits = set_quant_bits(k_important, k_general, k_unimportant, v_important, v_general, v_unimportant)
    return quant_bits


def plot_weight_heatmaps(model, num_layers, output_dir="weight_heatmaps_3d", sample_rate=4):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    darkblue = to_rgb("darkblue")
    lightgray = to_rgb("lightgray")
    darkred = to_rgb("darkred")

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [(0, darkblue), (0.12, darkblue), (0.12, darkred), (1, darkred)]
    )

    selected_layers = [0, 1, 11, 12, 30, 31]

    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(2, 6, wspace=0.3, hspace=0.1)

    for i, layer_idx in enumerate(selected_layers):

        weight = model.model.layers[layer_idx].self_attn.k_proj.weight.detach().cpu().numpy()
      
        weight_sampled = weight[::sample_rate, ::sample_rate]

        weight_abs = np.abs(weight_sampled)
    
        # log(1 + |w|)
        weight_log = np.log1p(weight_abs)

        weight_log_max = np.max(weight_log)
        weight_log_normalized = weight_log / weight_log_max if weight_log_max > 0 else weight_log

        token_dim, channel_dim = weight_sampled.shape
        tokens = np.arange(token_dim)
        channels = np.arange(channel_dim)
        T, C = np.meshgrid(tokens, channels, indexing='ij')

        ax = fig.add_subplot(gs[0, i], projection='3d')
        surf = ax.plot_surface(T, C, weight_log_normalized, cmap=custom_cmap, vmin=0, vmax=1)

        weight_norm = np.linalg.norm(weight)
        weight_min, weight_max = np.min(weight), np.max(weight)

        ax.set_title(
            f"Layer {layer_idx} - K\n"
            f"Norm: {weight_norm:.2f}, Range: [{weight_min:.2f}, {weight_max:.2f}]",
            fontsize=14
        )
        ax.set_xlabel("Token", fontsize=8)
        ax.set_ylabel("Channel", fontsize=8)
        ax.set_zlabel("Normalized Log-Value", fontsize=8)
        
        ax.view_init(elev=30, azim=-60)

    for i, layer_idx in enumerate(selected_layers):
        weight = model.model.layers[layer_idx].self_attn.v_proj.weight.detach().cpu().numpy()

        weight_sampled = weight[::sample_rate, ::sample_rate]

        weight_abs = np.abs(weight_sampled)

        weight_log = np.log1p(weight_abs)

        weight_log_max = np.max(weight_log)
        weight_log_normalized = weight_log / weight_log_max if weight_log_max > 0 else weight_log

        token_dim, channel_dim = weight_sampled.shape
        tokens = np.arange(token_dim)
        channels = np.arange(channel_dim)
        T, C = np.meshgrid(tokens, channels, indexing='ij')

        ax = fig.add_subplot(gs[1, i], projection='3d')
        surf = ax.plot_surface(T, C, weight_log_normalized, cmap=custom_cmap, vmin=0, vmax=1)

        weight_norm = np.linalg.norm(weight)
        weight_min, weight_max = np.min(weight), np.max(weight)

        ax.set_title(
            f"Layer {layer_idx} - V\n"
            f"Norm: {weight_norm:.2f}, Range: [{weight_min:.2f}, {weight_max:.2f}]",
            fontsize=14
        )
        ax.set_xlabel("Token", fontsize=8)
        ax.set_ylabel("Channel", fontsize=8)
        ax.set_zlabel("Normalized Log-Value", fontsize=8)

        ax.view_init(elev=30, azim=-60)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.0125, 0.7])
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    # cbar.set_label("Normalized Log-transformed Value", fontsize=15)

    cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.3, hspace=0.1)

    output_path = os.path.join(output_dir, "selected_layers_k_v_weights_3d_surface_log.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"3D surface plot (logarithmic transformation) saved to {output_path}")

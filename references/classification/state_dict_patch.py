import torch
from copy import deepcopy

eps = 1e-12

def need_to_preserve_head(layer_id, head_id, new_sd):
    for m in ["q", "k", "v"]:
        weight = new_sd[f"nncf_module.encoder.layers.encoder_layer_{layer_id}.self_attention.{m}_proj.weight"]
        if (weight[head_id*64 : (head_id+1)*64, :].abs() > eps).float().sum() > 0:
            return True
        bias = new_sd[f"nncf_module.encoder.layers.encoder_layer_{layer_id}.self_attention.{m}_proj.bias"]
        bias = bias.reshape(-1)
        if (bias[head_id*64 : (head_id+1)*64].abs() > eps).float().sum() > 0:
            return True
    m = 'out'
    weight = new_sd[f"nncf_module.encoder.layers.encoder_layer_{layer_id}.self_attention.{m}_proj.weight"]
    if (weight[:, head_id*64 : (head_id+1)*64].abs() > eps).float().sum() > 0:
        return True
    return False

def need_to_preserve_ffn(layer_id, ffn_id, new_sd):
    weight = new_sd[f'nncf_module.encoder.layers.encoder_layer_{layer_id}.mlp.0.weight']
    if (weight[ffn_id, :].abs() > eps).float().sum() > 0:
        return True
    weight = new_sd[f'nncf_module.encoder.layers.encoder_layer_{layer_id}.mlp.3.weight']
    if (weight[:, ffn_id].abs() > eps).float().sum() > 0:
        return True
    bias = new_sd[f'nncf_module.encoder.layers.encoder_layer_{layer_id}.mlp.0.bias']
    bias = bias.reshape(-1)
    if (bias[ffn_id].abs() > eps).float().sum() > 0:
        return True
    return False

def calc_sparsity(preserved_by_layer: dict):
    all_counts = 0
    preserved = 0
    for i, (preserve_heads, preserve_ffns) in preserved_by_layer.items():
        all_counts += 768 * 768 * 4 + 768 * 4
        preserved += 768 * 64 * len(preserve_heads) * 4 + 768 + 64 * len(preserve_heads) * 3
        all_counts += 768 * 3072 * 2 + 768 + 3072
        preserved += 768 * len(preserve_ffns) * 2 + len(preserve_ffns) + 768
    return (all_counts - preserved) / all_counts

def resolve_structured_mask(sd: dict):
    keys = []
    new_sd = deepcopy(sd)
    for i in range(12):
        for m in ["q", "k", "v", "out"]:
            keys.append(f"nncf_module.encoder.layers.encoder_layer_{i}.self_attention.{m}_proj")
        for m in [0, 3]:
            keys.append(f"nncf_module.encoder.layers.encoder_layer_{i}.mlp.{m}")

    for key in keys:
        for d in ["weight", "bias"]:
            importance = f"{key}.pre_ops.0.op._{d}_importance"
            new_sd[importance] = torch.ones_like(sd[importance]) * 100.0
            param = sd[f"{key}.{d}"] * sd[f"{key}.pre_ops.0.op.{d}_ctx._binary_mask"]
            param[param.abs() < eps] = 0.
            new_sd[f"{key}.{d}"] = param
            new_sd[f"{key}.pre_ops.0.op.{d}_ctx._binary_mask"] = torch.ones_like(sd[f"{key}.pre_ops.0.op.{d}_ctx._binary_mask"])
    
    preserved_by_layer = {}
    for layer_id in range(12):
        preserved_heads = []
        preserved_ffns = []
        for head_id in range(12):
            if need_to_preserve_head(layer_id, head_id, new_sd):
                preserved_heads.append(head_id)
        for ffn_id in range(768 * 4):
            if need_to_preserve_ffn(layer_id, ffn_id, new_sd):
                preserved_ffns.append(ffn_id)
        preserved_by_layer[layer_id] = (preserved_heads, preserved_ffns)

    return new_sd, preserved_by_layer


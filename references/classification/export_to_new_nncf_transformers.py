import shutil

import jstyleson
import torch
import torchvision
import transformers
import nncf
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
import logging
from nncf import set_log_level
set_log_level(logging.INFO)

# run with nncf=2.5

model = transformers.AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
nncf_config = {
    "input_info": [{"sample_size": [1, 3, 224, 224]}],
    "compression": [
        {
            "algorithm": "movement_sparsity",
            "params": {
                "warmup_start_epoch": 2,
                "warmup_end_epoch": 7,
                "importance_regularization_factor": 0.01,
                "enable_structured_masking": False,
            },
            "sparse_structure_by_scopes": [
                {"mode": "block", "sparse_factors": [32, 32], "target_scopes": "{re}ViTSelfAttention.*NNCFLinear"},
                {"mode": "block", "sparse_factors": [32, 32], "target_scopes": "{re}ViTSelfOutput.*NNCFLinear"},
                {"mode": "per_dim", "axis": 0, "target_scopes": "{re}ViTIntermediate"},
                {"mode": "per_dim", "axis": 1, "target_scopes": "{re}ViTOutput"},
            ],
            "ignored_scopes": [
                "{re}NNCFConv2d",
                "{re}Embedding",
                "{re}classifier",
            ],
        },
        {
            "algorithm": "quantization",
            "initializer": {
                "range": {
                    "num_init_samples": 4,
                    "type": "percentile",
                    "params": {"min_percentile": 0.01, "max_percentile": 99.99},
                },
                "batchnorm_adaptation": {"num_bn_adaptation_samples": 4},
            },
            "activations": {"mode": "symmetric"},
            "weights": {"mode": "symmetric", "signed": True, "per_channel": False},
        },
    ],
}
nncf_config["log_dir"] = "/tmp"



class RandDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return torch.randn((3, 224, 224)), 0


fake_data_loader = torch.utils.data.DataLoader(RandDataset(), batch_size=4)
nncf_config = NNCFConfig.from_dict(nncf_config)
nncf_config = register_default_init_args(
    nncf_config=nncf_config, train_loader=fake_data_loader
)  # TODO: distributed_callbacks and execution_parameters
print(nncf_config)

compression_ctrl, model = create_compressed_model(model, nncf_config)
mvmt_ctrl = compression_ctrl.child_ctrls[0]

names = []
for sinfo in mvmt_ctrl.sparsified_module_info:
    names.append(sinfo.module_node_name)
assert len(names) == 6 * 12


from state_dict_patch import resolve_structured_mask
new_sd, preserved_by_layer = resolve_structured_mask(
    torch.load(
        "/nvme2/yujiepan/workspace/jpqd-vit/LOGS/ww42/1011.6586-vit-jpqnd-wt0wr0.055-prune2to7f8-epo45lr5e-5wd1e-6_2card/model_40.pth"
    )["model"]
)

state_dict_to_mask = model.state_dict()
for i in range(12):
    # model weights
    for p_name in ["weight", "bias"]:
        state_dict_to_mask[f"vit.encoder.layer.{i}.attention.attention.query.{p_name}"] = new_sd[
            f"nncf_module.encoder.layers.encoder_layer_{i}.self_attention.q_proj.{p_name}"
        ]
        state_dict_to_mask[f"vit.encoder.layer.{i}.attention.attention.key.{p_name}"] = new_sd[
            f"nncf_module.encoder.layers.encoder_layer_{i}.self_attention.k_proj.{p_name}"
        ]
        state_dict_to_mask[f"vit.encoder.layer.{i}.attention.attention.value.{p_name}"] = new_sd[
            f"nncf_module.encoder.layers.encoder_layer_{i}.self_attention.v_proj.{p_name}"
        ]
        state_dict_to_mask[f"vit.encoder.layer.{i}.attention.output.dense.{p_name}"] = new_sd[
            f"nncf_module.encoder.layers.encoder_layer_{i}.self_attention.out_proj.{p_name}"
        ]
        state_dict_to_mask[f"vit.encoder.layer.{i}.intermediate.dense.{p_name}"] = new_sd[
            f"nncf_module.encoder.layers.encoder_layer_{i}.mlp.0.{p_name}"
        ]
        state_dict_to_mask[f"vit.encoder.layer.{i}.output.dense.{p_name}"] = new_sd[
            f"nncf_module.encoder.layers.encoder_layer_{i}.mlp.3.{p_name}"
        ]
    # weight_importance
    importance_keys = [key for key in state_dict_to_mask if key.endswith("_importance")]
    assert len(importance_keys) == 12 * (6 * 2)
    for key in importance_keys:
        state_dict_to_mask[key] = torch.ones_like(state_dict_to_mask[key]) * 100.0

    mask_keys = [key for key in state_dict_to_mask if key.endswith("binary_mask")]
    assert len(mask_keys) == 12 * (6 * 2)
    for key in mask_keys:
        state_dict_to_mask[key] = torch.ones_like(state_dict_to_mask[key])

model.load_state_dict(state_dict_to_mask)
compression_ctrl.export_model("/home/yujiepan/work2/jpqd-vit/LOGS/2023-re-export/hf-jpqd-synthetic.onnx")

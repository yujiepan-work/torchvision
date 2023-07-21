import torch
import nncf
import torchvision
import shutil
from nncf import NNCFConfig
from nncf.torch import register_default_init_args
from nncf.torch import create_compressed_model
import jstyleson

# run with nncf=2.5

model = torchvision.models.vit_b_16(weights=None, num_classes=1000)
with open(
    "/nvme2/yujiepan/workspace/jpqd-vit/LOGS/ww42/1011.6586-vit-jpqnd-wt0wr0.055-prune2to7f8-epo45lr5e-5wd1e-6_2card/vit_b16_jpqnd_2to7f8_wt0wr0.055_2card.ft.json",
    "r",
) as f:
    nncf_config = jstyleson.load(f)

nncf_config["log_dir"] = "/tmp"
override_qcfg_init = dict(range=dict(num_init_samples=0), batchnorm_adaptation=dict(num_bn_adaptation_samples=0))
if isinstance(nncf_config["compression"], list):
    for algo in nncf_config["compression"]:
        if algo["algorithm"] == "quantization":
            algo["initializer"].update(override_qcfg_init)
        if algo["algorithm"] == "movement_sparsity":
            algo["params"] = {
                "warmup_start_epoch": 1,
                "warmup_end_epoch": 4,
                "importance_regularization_factor": 0.01,
                "enable_structured_masking": False,
            }
            algo["sparse_structure_by_scopes"] = [
                {"mode": "block", "sparse_factors": [32, 32], "target_scopes": "{re}.*MultiHeadAttention*"},
                {"mode": "per_dim", "axis": 0, "target_scopes": "{re}.*MLPBlock.mlp./NNCFLinear.0"},
                {"mode": "per_dim", "axis": 1, "target_scopes": "{re}.*MLPBlock.mlp./NNCFLinear.3"},
            ]
elif nncf_config["compression"]["algorithm"] == "quantization":
    nncf_config["compression"]["initializer"].update(override_qcfg_init)


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

from state_dict_patch import resolve_structured_mask

new_sd, preserved_by_layer = resolve_structured_mask(
    torch.load(
        "/nvme2/yujiepan/workspace/jpqd-vit/LOGS/ww42/1011.6586-vit-jpqnd-wt0wr0.055-prune2to7f8-epo45lr5e-5wd1e-6_2card/model_40.pth"
    )["model"]
)

nnnew_sd = {}
for k, v in new_sd.items():
    if k.startswith("nncf_module."):
        k = k.replace("nncf_module.", "")
    if k.startswith("external_quantizers"):
        k = f"_nncf.{k}"
    if k.endswith("._weight_importance"):
        k = k.replace("._weight_importance", ".weight_importance")
    if k.endswith("._bias_importance"):
        k = k.replace("._bias_importance", ".bias_importance")
    if "/LayerNorm[" in k:
        k = k.replace("/LayerNorm[", "/NNCFLayerNorm[")

    nnnew_sd[k] = v

model.load_state_dict(nnnew_sd)
compression_ctrl.export_model("./model-manual.onnx")

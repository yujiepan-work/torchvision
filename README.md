This repo is for appying JPQD into ViT.

Setup:
 - See `CONFIGS/pip` and `CONIFIGS/conda`
 - NNCF version: `https://github.com/yujiepan-work/nncf/releases/tag/vit-jpqd-old-codes`
 - torchvision: this

See `CONFIGS/run2307.py` for re-evaluating models.

To export the onnx model:
The original onnx models are converted to IR using a custom branch: `https://github.com/daniil-lyakhov/openvino/tree/dl/pruning_transformers`. However, it will error if you use benchmark_app in ov version 2023.0.1.

I failed to export a 2023.0.1 compatible onnx. So I generate a synthetic model from huggingface's ViT, but the structured masks are transferred. To do this,
please re-install nncf=2.5.0 and transformers=4.31.0, them run `./references/classification/export_to_new_nncf_transformers.py`

For logs and pretrained models, see https://github.com/intel-sandbox/yujiepan.project.23h2.re-export-vit-models-from-torchvision-jpqd

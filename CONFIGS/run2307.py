from toytools.batchrun import Task, Launcher, avail_cuda_list
from toytools.misc import get_hash
import os
from pathlib import Path
import datetime

env = os.environ.copy()
env["YUJIE_SIGMOID_THRESHOLD"] = "0"
env["YUJIE_GLOBAL_LR"] = "0"

root = Path("/nvme2/yujiepan/workspace/jpqd-vit/training/torchvision/references/classification")
root_folder = Path("/nvme2/yujiepan/workspace/jpqd-vit/LOGS/2023-re-collect-summary-v2")

nncf_pths = []

# for epoch in [40, 41, 42, 43, 44]:
for epoch in [40]:
    nncf = "/nvme2/yujiepan/workspace/jpqd-vit/LOGS/ww42/1011.6586-vit-jpqnd-wt0wr0.055-prune2to7f8-epo45lr5e-5wd1e-6_2card/vit_b16_jpqnd_2to7f8_wt0wr0.055_2card.ft.json"
    pth = f"/nvme2/yujiepan/workspace/jpqd-vit/LOGS/ww42/1011.6586-vit-jpqnd-wt0wr0.055-prune2to7f8-epo45lr5e-5wd1e-6_2card/model_{epoch}.pth"
    nncf_pths.append((nncf, pth))

# for epoch in [40, 41, 42, 43, 44]:
#     nncf = "/home/yujiepan/work2/jpqd-vit/LOGS/ww42/1012.689a-vit-jpqnd-wt0wr0.060-prune2to7f8-epo45lr5e-5wd1e-6_2card/vit_b16_jpqnd_2to7f8_wt0wr0.060_2card.ft.json"
#     pth = f"/home/yujiepan/work2/jpqd-vit/LOGS/ww42/1012.689a-vit-jpqnd-wt0wr0.060-prune2to7f8-epo45lr5e-5wd1e-6_2card/model_{epoch}.pth"
#     nncf_pths.append((nncf, pth))

# for epoch in [40, 41, 42, 43, 44]:
#     nncf = "/home/yujiepan/work2/jpqd-vit/LOGS/ww42/1008.713e-vit-jpqnd-wt0wr0.045-prune2to7f8-epo45lr5e-5wd1e-6_2card/vit_b16_jpqnd_2to7f8_wt0wr0.045_2card.ft.json"
#     pth = f"/home/yujiepan/work2/jpqd-vit/LOGS/ww42/1008.713e-vit-jpqnd-wt0wr0.045-prune2to7f8-epo45lr5e-5wd1e-6_2card/model_{epoch}.pth"
#     nncf_pths.append((nncf, pth))

tasks = []
for nncf, pth in nncf_pths:
    folder = root_folder / ('export_' + get_hash(pth))
    cmd = f"""torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:0 --rdzv_backend=c10d train_notrain_202307.py \
        --model vit_b_16 \
        --epochs 15 \
        --batch-size 256 \
        --workers 16 \
        --opt adamw \
        --lr-scheduler cosineannealinglr \
        --lr-warmup-method linear \
        --lr-warmup-epochs 0 \
        --lr-warmup-decay 0.033 \
        --amp \
        --label-smoothing 0.11 \
        --mixup-alpha 0.2 \
        --auto-augment ra \
        --clip-grad-norm 1 \
        --ra-sampler \
        --cutmix-alpha 1.0 \
        --output-dir {folder} \
        --data-path /nvme1/datasets/ilsvrc2012/torchvision \
        --weights IMAGENET1K_V1  \
        --nncf_config {nncf} \
        --manual_load {pth}  """

    task = Task(
        cmd=cmd,
        cwd=root,
        io_folder=folder,
        env=env,
        cuda_quantity=1,
        identifier=pth[pth.index("LOGS") :],
    )
    tasks.append(task)

Launcher([1, 2, 3]).run(tasks, add_timestamp_to_log=False)

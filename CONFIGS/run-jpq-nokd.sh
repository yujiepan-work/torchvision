#!/usr/bin/env bash
export TZ=UTC-8
time=`date +"%m%d"`.$(printf "%04x"  $(($((  $(date +%s) - $(date +%s -d `date +%F`)  ))/2)))
echo $time

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
# export WANDB_USERNAME=
# export WANDB_API_KEY=
export WANDB_PROJECT="vit_b_16"

PROJROOT=$HOME/work2/jpqd-vit/
DATAPATH=/nvme1/datasets/ilsvrc2012/torchvision
CONDAROOT=$HOME/miniconda3
CONDAENV=jpqd-vit

WORKDIR=$PROJROOT/training/torchvision/references/classification
OUTROOT=$PROJROOT/LOGS/ww39

NNCFCFG=/home/yujiepan/work2/jpqd-vit/CONFIGS/nncf_configs/vit_b_16_jpq_2to5_wt0wr0.05.json
NNCFCFG=/home/yujiepan/work2/jpqd-vit/CONFIGS/nncf_configs/vit_b_16_jpq_0to2_wr0.1.json
# NNCFCFG=/home/yujiepan/work2/jpqd-vit/CONFIGS/nncf_configs/vit_b_16_jpq_r0.1.json
# NNCFCFG=/home/yujiepan/work2/jpqd-vit/CONFIGS/nncf_configs/vit_b_16_jpq_2to5_wt0wr0.05.json
# NNCFCFG=/home/yujiepan/work2/jpqd-vit/CONFIGS/nncf_configs/vit_b_16_jpq_0to3.json
NNCFCFG_DEBUG=$NNCFCFG
LR=1e-5
WD=1e-5
RUNID=$time-vit-jpqnokd-ct0wr0.1-prune0to2-epo2+12lr${LR}wd${WD}-fixfill
export YUJIE_SIGMOID_THRESHOLD=0
export YUJIE_GLOBAL_LR=1

if [[ $1 == "dryrun" ||  $1 == "debug" ]]; then
    OUTROOT=$OUTROOT/debug
    RUNID=$1-$RUNID
    export WANDB_DISABLED=false
    export WANDB_PROJECT="${WANDB_PROJECT}-debug"
    NNCFCFG=$NNCFCFG_DEBUG
fi

OUTDIR=$OUTROOT/$RUNID
mkdir -p $OUTDIR

cmd="train.py \
    --model vit_b_16 \
    --epochs 12 \
    --batch-size 128 \
    --workers 16 \
    --opt adamw \
    --lr $LR \
    --wd $WD \
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
    --output-dir $OUTDIR \
    --data-path $DATAPATH \
    --wandb_id $RUNID "
cmd="$cmd --weights IMAGENET1K_V1 "
cmd="$cmd --nncf_config $NNCFCFG "
# cmd="$cmd --manual_load /home/yujiepan/work2/jpqd-vit/LOGS/ptq_model/jpq-pytorch.v3-512.bin "
cmd="$cmd --manual_load /home/yujiepan/work2/jpqd-vit/LOGS/ww39/0921.0767-vit-jpqnokd-ct0wr0.05-prune2to5-epo12lr1e-5wd1e-5/model_1.pth "
# cmd="$cmd --manual_load  /home/yujiepan/work2/jpqd-vit/LOGS/ww38/0916.1d88-vit-jpqnokd-epo10lr1e-5wd1e-4warm0decay0.5/model_0.pth "
# cmd="$cmd  --manual_load /home/yujiepan/work2/jpqd-vit/LOGS/ww39/0920.53c0-vit-jpqnokd-wt0wr0.05-prune4to7-epo1+15lr1e-5wd1e-4/model_3.pth "

if [[ $2 == "" || $2 == "1" ]]; then
    cmd="python $cmd"
else
    echo "Using DDP with $2 cards." 
    cmd="torchrun --nproc_per_node=$2 --master_port=$(($RANDOM%10000+50000)) $cmd"
fi

cd $WORKDIR
source $CONDAROOT/etc/profile.d/conda.sh
# export CUDA_HOME=$CONDAROOT/envs/$CONDAENV
conda activate ${CONDAENV}
which python | tee $OUTDIR/python-env.log
printenv | sort 2<&1 > $OUTDIR/bash-env.log

if [[ $1 == "local" ]]; then
    cmd="nohup ${cmd}"
    echo "${cmd}" | tee $OUTDIR/cmd.log
    eval $cmd >> $OUTDIR/run.log 2>&1 &
elif [[ $1 == "dryrun" ]]; then
    cmd="${cmd} --max_steps 30 "
    echo "${cmd}" | tee $OUTDIR/cmd.log
    eval $cmd >> $OUTDIR/run.log 2>&1 &
elif [[ $1 == "debug" ]]; then
    cmd="${cmd} --max_steps 30 "
    echo "${cmd}" | tee $OUTDIR/cmd.log
    echo "### End of CMD" >> $OUTDIR/cmd.log
    eval $cmd 2>&1 | tee $OUTDIR/run.log
fi
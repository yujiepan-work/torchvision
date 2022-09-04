#!/usr/bin/env bash
export TZ=UTC-8
time=`date +"%m%d"`.$(printf "%04x"  $(($((  $(date +%s) - $(date +%s -d `date +%F`)  ))/2)))
echo $time

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
# export WANDB_USERNAME=
# export WANDB_API_KEY=
export WANDB_PROJECT="vit-dev"

PROJROOT=$HOME/work2/jpqd-vit
DATAPATH=/nvme1/datasets/ilsvrc2012/torchvision
CONDAROOT=$HOME/miniconda3
CONDAENV=jpqd-vit

WORKDIR=$PROJROOT/torchvision/references/classification
OUTROOT=$PROJROOT/LOGS/ww37

NNCFCFG=nncf_configs/vit_b_16_debug.json
NNCFCFG_DEBUG=$NNCFCFG
RUNID=$time-vit

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
    --epochs 300 \
    --batch-size 256 \
    --workers 8 \
    --opt adamw \
    --lr 0.003 \
    --wd 0.3 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-method linear \
    --lr-warmup-epochs 30 \
    --lr-warmup-decay 0.033 \
    --amp \
    --label-smoothing 0.11 \
    --mixup-alpha 0.2 \
    --auto-augment ra \
    --clip-grad-norm 1 \
    --ra-sampler \
    --cutmix-alpha 1.0 \
    --model-ema \
    --output-dir $OUTDIR \
    --data-path $DATAPATH \
    --wandb_id $RUNID "
cmd="$cmd --weights ViT_B_16_Weights.IMAGENET1K_V1 "
cmd="$cmd --nncf_config $NNCFCFG "
# cmd="$cmd --test-only"

if [[ $2 == "" || $2 == "1" ]]; then
    cmd="torchrun $cmd"
else
    echo "Using DDP with $2 cards." 
    cmd="torchrun --nproc_per_node=$2 --master_port=$(($RANDOM%10000+50000)) $cmd"
fi

cd $WORKDIR
source $CONDAROOT/etc/profile.d/conda.sh
conda activate ${CONDAENV}
which python | tee $OUTDIR/python-env.log
printenv | sort 2<&1 > $OUTDIR/bash-env.log

if [[ $1 == "local" ]]; then
    cmd="nohup ${cmd}"
    echo "${cmd}" | tee $OUTDIR/cmd.log
    eval $cmd >> $OUTDIR/run.log 2>&1 &
elif [[ $1 == "dryrun" ]]; then
    cmd="${cmd} --max_steps 50 "
    echo "${cmd}" | tee $OUTDIR/cmd.log
    eval $cmd >> $OUTDIR/run.log 2>&1 &
elif [[ $1 == "debug" ]]; then
    cmd="${cmd} --max_steps 50 "
    echo "${cmd}" | tee $OUTDIR/cmd.log
    echo "### End of CMD" >> $OUTDIR/cmd.log
    eval $cmd 2>&1 | tee $OUTDIR/run.log
fi
#!/usr/bin/env bash

# python wait_for_n_cards.py 2

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
OUTROOT=$PROJROOT/LOGS/ww42

NNCFCFG=/home/yujiepan/work2/jpqd-vit/CONFIGS/nncf_configs/vit_b16_jpqnd_2to7f8_wt0wr0.065_2card.ft.json
NNCFCFG_DEBUG=$NNCFCFG
LR=8e-5
WD=1e-6
EPOCH=45
RUNID=$time-vit-jpqnd-wt0wr0.065-prune2to7f8-epo${EPOCH}lr${LR}wd${WD}_2card
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
    --epochs  $EPOCH \
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
cmd="$cmd --teacher vit_b_16 "
cmd="$cmd --teacher_weights IMAGENET1K_V1 "
cmd="$cmd --nncf_config $NNCFCFG "

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
    eval $cmd 1> $OUTDIR/stdout.log 2>$OUTDIR/stderr.log &
elif [[ $1 == "dryrun" ]]; then
    cmd="${cmd} --max_steps 20 "
    echo "${cmd}" | tee $OUTDIR/cmd.log
    eval $cmd 1> $OUTDIR/stdout.log 2>$OUTDIR/stderr.log &
elif [[ $1 == "debug" ]]; then
    cmd="${cmd} --max_steps 20 "
    echo "${cmd}" | tee $OUTDIR/cmd.log
    echo "### End of CMD" >> $OUTDIR/cmd.log
    eval $cmd 2>&1 | tee $OUTDIR/run.log
fi
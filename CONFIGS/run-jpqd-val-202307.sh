#!/usr/bin/env bash
export TZ=UTC-8
time=`date +"%m%d"`.$(printf "%04x"  $(($((  $(date +%s) - $(date +%s -d `date +%F`)  ))/2)))
echo $time

PROJROOT=$HOME/work2/jpqd-vit/
DATAPATH=/nvme1/datasets/ilsvrc2012/torchvision
CONDAROOT=$HOME/miniconda3
CONDAENV=/home/yujiepan/miniconda3/envs/jpqd-vit

WORKDIR=$PROJROOT/training/torchvision/references/classification
OUTROOT=$PROJROOT/LOGS/2023-re-collect

NNCFCFG=/nvme2/yujiepan/workspace/jpqd-vit/LOGS/ww42/1012.689a-vit-jpqnd-wt0wr0.060-prune2to7f8-epo45lr5e-5wd1e-6_2card/vit_b16_jpqnd_2to7f8_wt0wr0.060_2card.ft.json

LR=3e-6
WD=1e-7
RUNID=$time-val_best_model
export YUJIE_SIGMOID_THRESHOLD=0
export YUJIE_GLOBAL_LR=0

if [[ $1 == "dryrun" ||  $1 == "debug" ]]; then
    OUTROOT=$OUTROOT/debug
    RUNID=$1-$RUNID
    export WANDB_DISABLED=false
    export WANDB_PROJECT="${WANDB_PROJECT}-debug"
    NNCFCFG=$NNCFCFG_DEBUG
fi

OUTDIR=$OUTROOT/$RUNID
mkdir -p $OUTDIR

cmd="train_notrain.py \
    --model vit_b_16 \
    --epochs 15 \
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
cmd="$cmd  --manual_load /nvme2/yujiepan/workspace/jpqd-vit/LOGS/ww42/1011.6586-vit-jpqnd-wt0wr0.055-prune2to7f8-epo45lr5e-5wd1e-6_2card/model_43.pth "

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
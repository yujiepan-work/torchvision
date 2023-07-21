#!/usr/bin/env bash
export TZ=UTC-8
time=`date +"%m%d"`.$(printf "%04x"  $(($((  $(date +%s) - $(date +%s -d `date +%F`)  ))/2)))
echo $time

export WANDB_DISABLED=true # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
# export WANDB_USERNAME=
# export WANDB_API_KEY=
export WANDB_PROJECT="vit_b_16"

PROJROOT=$HOME/work2/jpqd-vit/
DATAPATH=/nvme1/datasets/ilsvrc2012/torchvision
CONDAROOT=$HOME/miniconda3
CONDAENV=jpqd-vit

WORKDIR=$PROJROOT/training/torchvision/references/classification
OUTROOT=$PROJROOT/LOGS/ww40

NNCFCFG=/home/yujiepan/work2/jpqd-vit/LOGS/ww39/0925.6950-vit-jpqnd-wt0wr0.03-ft-epo15+15lr1e-5wd1e-5_2card/vit_b16_jpqnd_0to3_wt0wr0.03_1card.ft.json
# NNCFCFG=/home/yujiepan/work2/jpqd-vit/CONFIGS/nncf_configs/vit_b_16_jpq_r0.1.json
# NNCFCFG=/home/yujiepan/work2/jpqd-vit/CONFIGS/nncf_configs/vit_b_16_jpq_2to5_wt0wr0.05.json
# NNCFCFG=/home/yujiepan/work2/jpqd-vit/CONFIGS/nncf_configs/vit_b_16_jpq_0to3.json
NNCFCFG_DEBUG=$NNCFCFG
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
cmd="$cmd  --manual_load /home/yujiepan/work2/jpqd-vit/LOGS/ww39/0924.65e6-vit-jpqnd-wt0wr0.03-prune0to3add1-epo2+13lr3e-5wd1e-5_2card/model_3.pth "

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
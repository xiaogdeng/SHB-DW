#!/bin/bash

# model: ResNet18, ResNet34, MLP, Logistic
MODEL='Logistic'

# dataset: MNIST, CIFAR10
DATASET='MNIST'

LOGDIR_OUT='./log/'
LOGDIR='./runs/'
if [ ! -d $LOGDIR_OUT ]; then
  mkdir -p $LOGDIR_OUT
fi
if [ ! -d ${LOGDIR_OUT}/${MODEL}_${DATASET} ]; then
  mkdir -p ${LOGDIR_OUT}/${MODEL}_${DATASET}
fi


WEIGHT_DECAY=0.01
BATCH_SIZE=256
EPOCH=60
SEED=42
LR=0.01

ALPHA=0.999
BETA=0.999
BETA_MIN=0.1


OPTIMIZER='SHB_DW'
JOB_NAME=${LOGDIR_OUT}${MODEL}_${DATASET}/log_bs-${BATCH_SIZE}_lr-${LR}_wd-${WEIGHT_DECAY}_epoch-${EPOCH}_seed-${SEED}_optimizer-${OPTIMIZER}_beta-${BETA}_min-${BETA_MIN}_alpha-${ALPHA}
echo $JOB_NAME
python -u main.py --model $MODEL --dataset $DATASET --batch_size $BATCH_SIZE --logdir $LOGDIR \
                  --optimizer $OPTIMIZER --lr $LR --alpha $ALPHA  --beta $BETA --beta_min $BETA_MIN\
                  --epochs $EPOCH --seed $SEED --weight_decay $WEIGHT_DECAY  > ${JOB_NAME}.out 2>&1 &

BETA=0.9
OPTIMIZER='SHB'
JOB_NAME=${LOGDIR_OUT}${MODEL}_${DATASET}/log_bs-${BATCH_SIZE}_lr-${LR}_wd-${WEIGHT_DECAY}_epoch-${EPOCH}_seed-${SEED}_optimizer-${OPTIMIZER}_beta-${BETA}_min-${BETA_MIN}_alpha-${ALPHA}-2l
echo $JOB_NAME
python -u main.py --model $MODEL --dataset $DATASET --batch_size $BATCH_SIZE --logdir $LOGDIR \
                  --optimizer $OPTIMIZER --lr $LR --alpha $ALPHA  --beta $BETA --beta_min $BETA_MIN\
                  --epochs $EPOCH --seed $SEED --weight_decay $WEIGHT_DECAY  > ${JOB_NAME}.out 2>&1 &


OPTIMIZER='SGD'
JOB_NAME=${LOGDIR_OUT}${MODEL}_${DATASET}/log_bs-${BATCH_SIZE}_lr-${LR}_wd-${WEIGHT_DECAY}_epoch-${EPOCH}_seed-${SEED}_optimizer-${OPTIMIZER}_beta-${BETA}_min-${BETA_MIN}_alpha-${ALPHA}-2l
echo $JOB_NAME
python -u main.py --model $MODEL --dataset $DATASET --batch_size $BATCH_SIZE --logdir $LOGDIR \
                  --optimizer $OPTIMIZER --lr $LR --alpha $ALPHA  --beta $BETA --beta_min $BETA_MIN\
                  --epochs $EPOCH --seed $SEED --weight_decay $WEIGHT_DECAY  > ${JOB_NAME}.out 2>&1 &

LR=0.0001
OPTIMIZER='Adam'
JOB_NAME=${LOGDIR_OUT}${MODEL}_${DATASET}/log_bs-${BATCH_SIZE}_lr-${LR}_wd-${WEIGHT_DECAY}_epoch-${EPOCH}_seed-${SEED}_optimizer-${OPTIMIZER}_beta-${BETA}_min-${BETA_MIN}_alpha-${ALPHA}-2l
echo $JOB_NAME
python -u main.py --model $MODEL --dataset $DATASET --batch_size $BATCH_SIZE --logdir $LOGDIR \
                  --optimizer $OPTIMIZER --lr $LR --alpha $ALPHA  --beta $BETA --beta_min $BETA_MIN\
                  --epochs $EPOCH --seed $SEED --weight_decay $WEIGHT_DECAY  > ${JOB_NAME}.out 2>&1 &


echo 'finish!'


DATASET=$1
CLASS_IDX=$2
INSTANCE_IDX=$3

### Arugments ###

FEWSHOT_SEED="seed0"
N_SHOT=16


if [ $METHOD == "loft" ]; then
    BS=1
    NUM_TRAIN_EPOCH=600
    RANK=2
    LR=1e-3
    LR_WARMUP_STEP=10
    TTE=False
    CENTER_CROP=True

else # datadream
    BS=8
    NUM_TRAIN_EPOCH=200
    RANK=16
    LR=1e-4
    LR_WARMUP_STEP=100
    TTE=True
    CENTER_CROP=False

fi



### Run ###

accelerate launch main.py \
--dataset=$DATASET \
--train_batch_size=$BS \
--fewshot_seed=$FEWSHOT_SEED \
--learning_rate=$LR \
--lr_warmup_steps=$LR_WARMUP_STEP \
--num_train_epochs=$NUM_TRAIN_EPOCH \
--rank=$RANK \
--n_shot=$N_SHOT \
--target_class_idx=$CLASS_IDX \
--train_text_encoder=$TTE \
--center_crop=$CENTER_CROP \
--instance_idx=$INSTANCE_IDX \
--is_tqdm=True


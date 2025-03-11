
DATASET=$1
METHOD=$2


### Arugments ###

N_SHOT=16
FEWSHOT_SEED=seed0
NIPC=500 # number of images to be generated for each class

MODEL_TYPE=clip
BS=256
EPOCH=60
WARMUP_EPOCH=6
LR=1e-4
MIN_LR=1e-8
WD=1e-4
IS_MIX_AUG=True
EVAL_ITERATION=312


if [ $METHOD == "loft" ]; then
    LIW=0.5 # loft interpolation weight

else # datadream
    LIW=0

fi


### Run ###

python main.py \
--method=$METHOD \
--loft_interpolation_weight=$LIW \
--dataset=$DATASET \
--n_img_per_cls=$NIPC \
--n_shot=$N_SHOT \
--fewshot_seed=$FEWSHOT_SEED \
--model_type=$MODEL_TYPE \
--batch_size_per_gpu=$BS \
--epochs=$EPOCH \
--warmup_epochs=$WARMUP_EPOCH \
--lr=$LR \
--min_lr=$MIN_LR \
--wd=$WD \
--is_mix_aug=$IS_MIX_AUG \
--eval_iteration=$EVAL_ITERATION \
--save_model=False \
--is_distributed=False \
--is_set_log=False

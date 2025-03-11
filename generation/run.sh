
DATASET=$1
METHOD=$2


### Arugments ###

N_SHOT=16
FEWSHOT_SEED=seed0
NIPC=500 # number of images to be generated for each class
BS=5

if [ $METHOD == "loft" ]; then
    LIW=0.5 # loft interpolation weight

else # datadream
    LIW=0

fi


### Run ###

python main.py \
--dataset=$DATASET \
--method=$METHOD \
--n_shot=$N_SHOT \
--fewshot_seed=$FEWSHOT_SEED \
--loft_interpolation_weight=$LIW \
--n_image_per_class=$NIPC \
--bs=$BS \
--is_tqdm=True


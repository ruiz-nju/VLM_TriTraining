DEVICE=3


# for dataset in cifar10 cifar100 imagenet100
# do 
#     for seed in 1 2 3
#     do 
#         sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed $DEVICE
#         sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed $DEVICE
#         sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed $DEVICE
#         sh scripts/tritraining/base2novel_test_all.sh $dataset 16 0 $seed $DEVICE
#     done
# done

# DATASET=$1
# SEED=$2

# CFG=vit_b16
# SHOTS=$3
# SUB=$4
# DEVICE=$5

for seed in 1 2 3
do
    sh scripts/promptsrc/base2new_train.sh cifar10 $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar10 $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar10 $seed 16 new  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar10 $seed 16 all  $DEVICE

    sh scripts/promptsrc/base2new_train.sh cifar100 $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar100 $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar100 $seed 16 new  $DEVICE
    sh scripts/promptsrc/base2new_test.sh cifar100 $seed 16 all  $DEVICE

    sh scripts/promptsrc/base2new_train.sh imagenet100 $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh imagenet100 $seed 16 base  $DEVICE
    sh scripts/promptsrc/base2new_test.sh imagenet100 $seed 16 new  $DEVICE
    sh scripts/promptsrc/base2new_test.sh imagenet100 $seed 16 all  $DEVICE
done
DEVICE=3


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
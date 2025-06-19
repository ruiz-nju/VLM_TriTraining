DEVICE=3


for seed in 1 2 3
do
    # sh scripts/coop/ow_train.sh cifar10 16 base $seed $DEVICE
    sh scripts/coop/ow_test.sh cifar10 16 base $seed $DEVICE
    sh scripts/coop/ow_test.sh cifar10 16 new $seed $DEVICE
    sh scripts/coop/ow_test.sh cifar10 16 all $seed $DEVICE

    # sh scripts/coop/ow_train.sh cifar100 16 base $seed $DEVICE
    sh scripts/coop/ow_test.sh cifar100 16 base $seed $DEVICE
    sh scripts/coop/ow_test.sh cifar100 16 new $seed $DEVICE
    sh scripts/coop/ow_test.sh cifar100 16 all $seed $DEVICE

    # sh scripts/coop/ow_train.sh imagenet100 16 base $seed $DEVICE
    sh scripts/coop/ow_test.sh imagenet100 16 base $seed $DEVICE
    sh scripts/coop/ow_test.sh imagenet100 16 new $seed $DEVICE
    sh scripts/coop/ow_test.sh imagenet100 16 all $seed $DEVICE
done
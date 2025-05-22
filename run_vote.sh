DEVICE=1
for dataset in cifar10 cifar100 
do 
    for seed in 1 2 3
    do 
        sh scripts/vote/base2novel_train.sh $dataset 16 $seed $DEVICE
        sh scripts/vote/base2novel_test_base.sh $dataset 16 $seed $DEVICE
        sh scripts/vote/base2novel_test_new.sh $dataset 16 $seed $DEVICE
    done
done
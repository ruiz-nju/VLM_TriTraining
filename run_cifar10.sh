DEVICE=1


for dataset in cifar10
do 
    for seed in 1 2 3
    do 
        sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_all.sh $dataset 16 0 $seed $DEVICE
    done
done
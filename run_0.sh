DEVICE=0
for dataset in imagenet
do 
    for seed in 1 2 3
    do 
        sh scripts/tritraining/base2novel_train_new.sh $dataset 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed $DEVICE 
        sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed $DEVICE
    done
done

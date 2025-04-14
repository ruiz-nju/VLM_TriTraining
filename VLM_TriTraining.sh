DEVICE=1

for dataset in oxford_flowers  
do 
    for seed in 2
    do 
        sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed $DEVICE
    done
done

DEVICE=1

for dataset in dtd 
do 
    for seed in 1 
    do 
        sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed $DEVICE
    done
done
# for dataset in eurosat dtd fgvc_aircraft
# do 
#     for seed in 1 2 3
#     do 
#         sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed $DEVICE
#         sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed $DEVICE
#         sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed $DEVICE
#     done
# done
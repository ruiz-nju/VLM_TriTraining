DEVICE=3
# caltech101 dtd fgvc_aircraft
for dataset in food101
do 
    for seed in 1 2 3
    do 
        sh scripts/tritraining/base2novel_train.sh $dataset PromptSRC 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_base.sh $dataset PromptSRC 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_new.sh $dataset PromptSRC 16 0 $seed $DEVICE
    done
done


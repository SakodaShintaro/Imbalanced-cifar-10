for i in 5000 2500 500
do
    echo ${i}
    ./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=${i}
    cp -r ../result ../results/imbalanced${i}_normal

    ./train.py --coefficient_of_mse=1 --data_num_of_imbalanced_class=${i}
    cp -r ../result ../results/imbalanced${i}_with_ae

    ./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=${i} --copy_imbalanced_class
    cp -r ../result ../results/imbalanced${i}_with_copy

    ./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=${i} --use_mixup
    cp -r ../result ../results/imbalanced${i}_use_mixup

    ./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=${i} --use_prioritized_dataset
    cp -r ../result ../results/imbalanced${i}_use_prioritized_dataset
done
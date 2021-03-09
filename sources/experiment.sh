./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=5000
cp -r ../result ../results/full_data_normal

./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=5000 --use_mixup
cp -r ../result ../results/full_data_use_mixup

./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=5000 --use_prioritized_dataset
cp -r ../result ../results/full_data_use_prioritized_dataset

./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=2500
cp -r ../result ../results/imbalanced2500_normal

./train.py --coefficient_of_mse=1 --data_num_of_imbalanced_class=2500
cp -r ../result ../results/imbalanced2500_with_ae

./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=2500 --use_mixup
cp -r ../result ../results/imbalanced2500_use_mixup

./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=2500 --use_prioritized_dataset
cp -r ../result ../results/imbalanced2500_use_prioritized_dataset

./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=500
cp -r ../result ../results/imbalanced500_normal

./train.py --coefficient_of_mse=1 --data_num_of_imbalanced_class=500
cp -r ../result ../results/imbalanced500_with_ae

./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=500 --use_mixup
cp -r ../result ../results/imbalanced500_use_mixup

./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=500 --use_prioritized_dataset
cp -r ../result ../results/imbalanced500_use_prioritized_dataset

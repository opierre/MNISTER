#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2

##############################################
#			  Executions on GPU              #
##############################################
printf "\nExecutions on GPU\n"

# Without Batchs
cd ..
python3 ./mnist_load.py --model_version=1 --digits_per_img=True --nb_images=1 --print_prob=False >> ./GPU_results.txt 
echo "python3 ./mnist_load.py --model_version=1 --digits_per_img=True --nb_images=1 --print_prob=False"
# With Batchs
BATCHS_MAX=19

for i in $(seq 1 $BATCHS_MAX)
do 
	python3 ./mnist_load.py --model_version=1 --digits_per_img=True --nb_images=1 --print_prob=False --nb_batchs=$i >> ./GPU_results.txt 
	echo "python3 ./mnist_load.py --model_version=1 --digits_per_img=True --nb_images=1 --print_prob=False --nb_batchs="$i
done

##############################################
#			  Executions on CPU              #
##############################################
printf "\nExecutions on CPU\n"

# Without Batchs
CUDA_VISIBLE_DEVICES="" python3 ./mnist_load.py --model_version=1 --digits_per_img=True --nb_images=1 --print_prob=False >> ./CPU_results.txt 
echo "CUDA_VISIBLE_DEVICES="" python3 ./mnist_load.py --model_version=1 --digits_per_img=True --nb_images=1 --print_prob=False"

# With Batchs
for i in $(seq 1 $BATCHS_MAX)
do 
	CUDA_VISIBLE_DEVICES="" python3 ./mnist_load.py --model_version=1 --digits_per_img=True --nb_images=1 --print_prob=False --nb_batchs=$i >> ./CPU_results.txt 
	echo "CUDA_VISIBLE_DEVICES="" python3 ./mnist_load.py --model_version=1 --digits_per_img=True --nb_images=1 --print_prob=False --nb_batchs="$i
done

#!/bin/bash

for i in {1..5}
do
    echo "Running iteration $i for dataset COX2"
    nohup python RTpool_compound.py --dataset COX2 --device 1 --num_pooling_layers 1&
    wait
done

for i in {1..5}
do
    echo "Running iteration $i for dataset BZR"
    nohup python RTpool_compound.py --dataset BZR --device 1 --num_pooling_layers 1 &
    wait
done
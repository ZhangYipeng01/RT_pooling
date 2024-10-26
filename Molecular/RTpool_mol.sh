#!/bin/bash

for i in {1..5}
do
    echo "Running iteration $i for dataset PTC_MR"
    nohup python RTpool_molecular.py --dataset PTC_MR --device 1 --lr 0.0002 --final_dropout 0.3 --num_pooling_layers 1&
    wait
done

for i in {1..5}
do
    echo "Running iteration $i for dataset PTC_MM"
    nohup python RTpool_molecular.py --dataset PTC_MM --device 1 --lr 0.0002 --final_dropout 0.3 &
    wait
done

for i in {1..5}
do
    echo "Running iteration $i for dataset PTC_FR"
    nohup python RTpool_molecular.py --dataset PTC_FR --device 1 --lr 0.0002 --final_dropout 0.3 &
    wait
done

for i in {1..5}
do
    echo "Running iteration $i for dataset PTC_FM"
    nohup python RTpool_molecular.py --dataset PTC_FM --device 1 --lr 0.0002 --final_dropout 0.3&
    wait
done

for i in {1..5}
do
    echo "Running iteration $i for dataset MUTAG"
    nohup python RTpool_molecular.py --dataset MUTAG --device 1 --num_pooling_layers 1&
    wait
done

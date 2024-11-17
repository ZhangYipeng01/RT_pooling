#!/bin/bash

nohup python get_rhomboid.py --dataset COX2&
wait
nohup python Dataset_RT_pooling.py --dataset COX2&
wait

nohup python get_rhomboid.py --dataset BZR&
wait
nohup python Dataset_RT_pooling.py --dataset BZR&
wait

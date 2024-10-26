#!/bin/bash

nohup python get_rhomboid_fromSMILES.py --dataset COX2&
wait
nohup python Dataset_RT_pooling.py --dataset COX2&
wait

nohup python get_rhomboid_fromSMILES.py --dataset BZR&
wait
nohup python Dataset_RT_pooling.py --dataset BZR&
wait
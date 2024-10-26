#!/bin/bash

nohup python get_rhomboid_fromSMILES.py --dataset PTC_MR&
wait
nohup python Dataset_RT_pooling.py --dataset PTC_MR&
wait

nohup python get_rhomboid_fromSMILES.py --dataset PTC_MM&
wait
nohup python Dataset_RT_pooling.py --dataset PTC_MM&
wait

nohup python get_rhomboid_fromSMILES.py --dataset PTC_FR&
wait
nohup python Dataset_RT_pooling.py --dataset PTC_FR&
wait

nohup python get_rhomboid_fromSMILES.py --dataset PTC_FM&
wait
nohup python Dataset_RT_pooling.py --dataset PTC_FM&
wait

nohup python get_rhomboid_fromSMILES.py --dataset MUTAG&
wait
nohup python Dataset_RT_pooling.py --dataset MUTAG&
wait

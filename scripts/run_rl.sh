#!/bin/bash

mkdir -p exp_res
# export HYDRA_FULL_ERROR=1  # for debug

EXPNAME="${EXPNAME:-ca_ionic_conductor}"
MODEL="${MODEL:-mattergen_ionic}"
REWARD="${REWARD:-ionic_conductor}"
DEVICE="${DEVICE:-cuda:0}"
CHEMICAL_SYSTEM="${CHEMICAL_SYSTEM:-Ca-P-S}"

nohup python -u main.py \
    "expname=${EXPNAME}" \
    pipeline=mat_invent \
    "model=${MODEL}" \
    "reward=${REWARD}" \
    logger=csv \
    "device=${DEVICE}" \
    "model.sample_cfg.properties_to_condition_on.chemical_system=${CHEMICAL_SYSTEM}" \
    > "exp_res/${EXPNAME}.log" 2>&1 &

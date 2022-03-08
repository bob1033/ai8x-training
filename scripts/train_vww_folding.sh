#!/bin/sh
python train.py --epochs 100 --optimizer Adam --lr 0.0001 --deterministic --compress schedule-vww.yaml --model wakeModel_deep --use-bias --dataset vww_folding --confusion --param-hist --embedding --device MAX78002 --qat-policy qat_policy_vww.yaml "$@"

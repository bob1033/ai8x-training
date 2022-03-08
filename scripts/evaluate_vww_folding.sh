#!/bin/sh
python train.py --model wakemodel_deep --dataset vww_folding --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/vww_folding_qat-q.pth.tar -8 --device MAX78002 "$@"
#python train.py --model wakemodel_deep --dataset vww_folding --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/wakemodel_qat.pth.tar --device MAX78002 --use-bias "$@"

#!/bin/bash

cd ..

# compatibility field; dabench resolves actual dataset paths from its config
DATA=/unused/with-dabench
TRAINER=DAMP

DATASET=visda17 # name of the dataset
CFG=damp  # config file
TAU=0.5 # pseudo label threshold
U=2.0 # coefficient for loss_u
SEED=1
RUN_TAG=${TAU}_${U}

NAME=sr
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains synthetic --target-domains real --seed ${SEED} TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

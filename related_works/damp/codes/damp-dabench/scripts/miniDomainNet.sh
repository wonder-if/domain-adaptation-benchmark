#!/bin/bash

cd ..

# compatibility field; dabench resolves actual dataset paths from its config
DATA=/unused/with-dabench
TRAINER=DAMP

DATASET=miniDomainNet # dataset config name
CFG=damp  # config file
TAU=0.5 # pseudo label threshold
U=1.0 # coefficient for loss_u
SEED=1
RUN_TAG=${TAU}_${U}

NAME=cp
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains painting --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=cr
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains real --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=cs
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains sketch --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=pc
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA}  --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains painting --target-domains clipart --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=pr
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA}  --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains painting --target-domains real --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=ps
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA}  --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains painting --target-domains sketch --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=rc
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA}  --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real --target-domains clipart --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=rp
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real --target-domains painting --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=rs
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real --target-domains sketch --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=sc
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains sketch --target-domains clipart --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=sp
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains sketch --target-domains painting --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=sr
DIR=output/${DATASET}/${TRAINER}/${CFG}/${RUN_TAG}_${NAME}/seed_${SEED}
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains sketch --target-domains real --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

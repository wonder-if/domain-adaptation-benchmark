# TASC office UniDA Results

- method: TASC
- backbone: CLIP ViT-B/16
- primary_metric: H-score for ODA/OPDA; accuracy for CDA/PDA
- aggregation: final metric snapshot per selected run
- num_runs: 24
- source: related_works/tasc/codes/tasc-dabench/output

## CDA

- primary_metric: accuracy
- num_runs: 6

| A->D | A->W | D->A | D->W | W->A | W->D | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| 90.6 | 89.4 | 82.2 | 97.1 | 82.6 | 99.4 | 90.2 |

## PDA

- primary_metric: accuracy
- num_runs: 6

| A->D | A->W | D->A | D->W | W->A | W->D | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| 100.0 | 100.0 | 84.6 | 100.0 | 85.0 | 100.0 | 94.9 |

## ODA

- primary_metric: H-score
- num_runs: 6

| A->D | A->W | D->A | D->W | W->A | W->D | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| 93.8 | 93.2 | 89.0 | 97.8 | 87.8 | 99.5 | 93.5 |

Auxiliary open-set metrics:

| Task | OS* | UNK | OS | AUROC | UCR |
| --- | --- | --- | --- | --- | --- |
| A->D | 99.3 | 88.8 | 98.4 | 94.3 | 94.2 |
| A->W | 99.5 | 87.6 | 98.4 | 97.2 | 97.2 |
| D->A | 80.7 | 99.1 | 82.4 | 97.4 | 93.2 |
| D->W | 99.0 | 96.6 | 98.8 | 99.7 | 99.7 |
| W->A | 78.6 | 99.4 | 80.5 | 97.6 | 93.5 |
| W->D | 100.0 | 98.9 | 99.9 | 100.0 | 100.0 |

## OPDA

- primary_metric: H-score
- num_runs: 6

| A->D | A->W | D->A | D->W | W->A | W->D | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| 96.4 | 82.3 | 80.3 | 90.0 | 81.9 | 90.7 | 86.9 |

Auxiliary open-set metrics:

| Task | OS* | UNK | OS | AUROC | UCR |
| --- | --- | --- | --- | --- | --- |
| A->D | 99.3 | 93.6 | 98.8 | 99.3 | 99.3 |
| A->W | 99.0 | 70.4 | 96.4 | 94.7 | 94.7 |
| D->A | 84.0 | 77.0 | 83.4 | 89.1 | 79.7 |
| D->W | 98.6 | 82.8 | 97.1 | 93.7 | 93.7 |
| W->A | 81.6 | 82.2 | 81.6 | 91.2 | 82.3 |
| W->D | 99.3 | 83.5 | 97.9 | 94.0 | 93.9 |

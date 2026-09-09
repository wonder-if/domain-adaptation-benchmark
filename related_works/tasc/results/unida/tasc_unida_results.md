# TASC UniDA Results

- method: TASC
- backbone: CLIP ViT-B/16
- primary_metric: H-score for ODA/OPDA; accuracy for CDA/PDA
- aggregation: final metric snapshot per selected run
- num_runs: 79
- source: related_works/tasc/codes/tasc-dabench/output
- selection: group by dataset, category-shift setting, and transfer task; keep the run with the largest final iteration

## Overview

| Dataset | Setting | Primary metric | Num runs | Avg |
| --- | --- | --- | --- | --- |
| office | CDA | accuracy | 6 | 90.2 |
| office | PDA | accuracy | 6 | 94.9 |
| office | ODA | H-score | 6 | 93.5 |
| office | OPDA | H-score | 6 | 86.9 |
| office-home | CDA | accuracy | 12 | 86.7 |
| office-home | PDA | accuracy | 12 | 89.8 |
| office-home | ODA | H-score | 12 | 84.1 |
| office-home | OPDA | H-score | 12 | 88.8 |
| domainnet | OPDA | H-score | 6 | 73.2 |
| visda-2017 | OPDA | H-score | 1 | 89.8 |

## office

### CDA

- primary_metric: accuracy
- num_runs: 6

| A->D | A->W | D->A | D->W | W->A | W->D | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| 90.6 | 89.4 | 82.2 | 97.1 | 82.6 | 99.4 | 90.2 |

### PDA

- primary_metric: accuracy
- num_runs: 6

| A->D | A->W | D->A | D->W | W->A | W->D | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| 100.0 | 100.0 | 84.6 | 100.0 | 85.0 | 100.0 | 94.9 |

### ODA

- primary_metric: H-score
- num_runs: 6

| A->D | A->W | D->A | D->W | W->A | W->D | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| 93.8 | 93.2 | 89.0 | 97.8 | 87.8 | 99.5 | 93.5 |

### OPDA

- primary_metric: H-score
- num_runs: 6

| A->D | A->W | D->A | D->W | W->A | W->D | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| 96.4 | 82.3 | 80.3 | 90.0 | 81.9 | 90.7 | 86.9 |

## office-home

### CDA

- primary_metric: accuracy
- num_runs: 12

| A->C | A->P | A->R | C->A | C->P | C->R | P->A | P->C | P->R | R->A | R->C | R->P | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 74.9 | 92.8 | 92.0 | 87.1 | 92.0 | 91.7 | 86.8 | 75.1 | 92.3 | 87.6 | 75.5 | 92.9 | 86.7 |

### PDA

- primary_metric: accuracy
- num_runs: 12

| A->C | A->P | A->R | C->A | C->P | C->R | P->A | P->C | P->R | R->A | R->C | R->P | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 81.8 | 94.3 | 93.9 | 90.3 | 94.0 | 93.4 | 90.8 | 78.7 | 93.6 | 90.7 | 83.5 | 93.1 | 89.8 |

### ODA

- primary_metric: H-score
- num_runs: 12

| A->C | A->P | A->R | C->A | C->P | C->R | P->A | P->C | P->R | R->A | R->C | R->P | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 77.4 | 87.7 | 89.8 | 81.6 | 90.0 | 87.5 | 82.3 | 78.2 | 88.0 | 81.2 | 77.7 | 87.5 | 84.1 |

### OPDA

- primary_metric: H-score
- num_runs: 12

| A->C | A->P | A->R | C->A | C->P | C->R | P->A | P->C | P->R | R->A | R->C | R->P | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 83.7 | 89.8 | 92.2 | 90.3 | 92.4 | 92.0 | 88.4 | 82.0 | 93.3 | 88.8 | 82.9 | 89.4 | 88.8 |

## domainnet

### OPDA

- primary_metric: H-score
- num_runs: 6

| Pnt->Rel | Pnt->Skt | Rel->Pnt | Rel->Skt | Skt->Pnt | Skt->Rel | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| 80.8 | 69.8 | 69.7 | 69.0 | 69.4 | 80.8 | 73.2 |

## visda-2017

### OPDA

- primary_metric: H-score
- num_runs: 1

| S->R | Avg |
| --- | --- |
| 89.8 | 89.8 |

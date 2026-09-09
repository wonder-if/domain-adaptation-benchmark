# TASC visda-2017 UniDA Results

- method: TASC
- backbone: CLIP ViT-B/16
- primary_metric: H-score for ODA/OPDA; accuracy for CDA/PDA
- aggregation: final metric snapshot per selected run
- num_runs: 1
- source: related_works/tasc/codes/tasc-dabench/output

## OPDA

- primary_metric: H-score
- num_runs: 1

| S->R | Avg |
| --- | --- |
| 89.8 | 89.8 |

Auxiliary open-set metrics:

| Task | OS* | UNK | OS | AUROC | UCR |
| --- | --- | --- | --- | --- | --- |
| S->R | 88.7 | 91.0 | 89.0 | 95.5 | 87.5 |

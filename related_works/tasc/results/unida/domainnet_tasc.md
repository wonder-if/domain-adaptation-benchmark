# TASC domainnet UniDA Results

- method: TASC
- backbone: CLIP ViT-B/16
- primary_metric: H-score for ODA/OPDA; accuracy for CDA/PDA
- aggregation: final metric snapshot per selected run
- num_runs: 6
- source: related_works/tasc/codes/tasc-dabench/output

## OPDA

- primary_metric: H-score
- num_runs: 6

| Pnt->Rel | Pnt->Skt | Rel->Pnt | Rel->Skt | Skt->Pnt | Skt->Rel | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| 80.8 | 69.8 | 69.7 | 69.0 | 69.4 | 80.8 | 73.2 |

Auxiliary open-set metrics:

| Task | OS* | UNK | OS | AUROC | UCR |
| --- | --- | --- | --- | --- | --- |
| Pnt->Rel | 77.9 | 83.9 | 77.9 | 91.3 | 81.9 |
| Pnt->Skt | 66.1 | 73.8 | 66.2 | 83.9 | 67.3 |
| Rel->Pnt | 64.1 | 76.2 | 64.2 | 83.7 | 68.1 |
| Rel->Skt | 65.6 | 72.9 | 65.6 | 84.3 | 66.5 |
| Skt->Pnt | 62.3 | 78.2 | 62.4 | 84.4 | 69.2 |
| Skt->Rel | 79.1 | 82.7 | 79.1 | 91.5 | 82.3 |

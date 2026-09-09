# DAMP UDA Results

## office-home (RN50)

- method: DAMP
- backbone: RN50
- primary_metric: accuracy
- metric_source: best
- num_runs: 12

| Ar->Cl | Ar->Pr | Ar->Rw | Cl->Ar | Cl->Pr | Cl->Rw | Pr->Ar | Pr->Cl | Pr->Rw | Rw->Ar | Rw->Cl | Rw->Pr | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 59.1 | 88.7 | 86.3 | 76.6 | 88.7 | 86.5 | 76.6 | 59.7 | 87.3 | 77.5 | 61.0 | 89.8 | 78.1 |

## minidomainnet (RN50)

- method: DAMP
- backbone: RN50
- primary_metric: accuracy
- metric_source: best
- num_runs: 12

| Tgt\Src | Clp | Pnt | Rel | Skt | Avg |
| --- | --- | --- | --- | --- | --- |
| Clp | - | 75.2 | 76.7 | 76.5 | 76.1 |
| Pnt | 74.9 | - | 74.8 | 74.8 | 74.8 |
| Rel | 87.0 | 87.1 | - | 87.5 | 87.2 |
| Skt | 71.0 | 69.4 | 69.5 | - | 70.0 |
| Avg | 77.6 | 77.2 | 73.7 | 79.6 | 77.0 |

## minidomainnet (ViT-B/16)

- method: DAMP
- backbone: ViT-B/16
- primary_metric: accuracy
- metric_source: best
- num_runs: 12

| Tgt\Src | Clp | Pnt | Rel | Skt | Avg |
| --- | --- | --- | --- | --- | --- |
| Clp | - | 88.9 | 88.1 | 89.0 | 88.7 |
| Pnt | 85.1 | - | 83.3 | 83.0 | 83.8 |
| Rel | 92.1 | 92.4 | - | 92.2 | 92.2 |
| Skt | 84.0 | 83.5 | 84.6 | - | 84.0 |
| Avg | 87.1 | 88.3 | 85.3 | 88.1 | 87.2 |

## visda-2017 (ViT-B/16)

- method: DAMP
- backbone: ViT-B/16
- primary_metric: average_class_accuracy
- metric_source: best
- num_runs: 1

| plane | bicycle | bus | car | horse | knife | mcycl | person | plant | sktbrd | train | truck | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 99.1 | 92.8 | 91.8 | 74.8 | 98.6 | 96.4 | 95.0 | 82.7 | 93.0 | 94.9 | 95.4 | 69.6 | 90.3 |

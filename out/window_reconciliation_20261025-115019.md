# Window-count reconciliation

| split | dataset-table windows (60 s at effective fps) | evaluation windows (1794 frames) | evaluation valid windows (stored) | evaluation valid windows (recomputed) | clips whose counts differ |
|---|---|---|---|---|---|
| val | 1202 | 1240 | 1239 | 1239 | 1 |

Clips whose dataset-table count differs from the evaluation (val):

| clip | effective fps | frames (table) | frames (evaluation) | valid (table) | valid (evaluation, stored) |
|---|---|---|---|---|---|
| mcd:8772_before | 18.41 | 1104 | 1794 | 52 | 89 |

| test | 1197 | 1202 | 1198 | 1198 | 1 |

Clips whose dataset-table count differs from the evaluation (test):

| clip | effective fps | frames (table) | frames (evaluation) | valid (table) | valid (evaluation, stored) |
|---|---|---|---|---|---|
| mcd:5223_before | 29.27 | 1756 | 1794 | 51 | 52 |


### Held-out nominal, 100 paired episodes (conditions: mean over 6 seeds ± 95 % t half-width)

| Controller | Margin (EUR/shift) | Safety | Delivery | Quality | Feasible | Dev margin, final checkpoint |
|---|---|---|---|---|---|---|
| uniform | -86.6 | 100 % | 0 % | 0 % | no | |
| fixed: Conservative | -13.2 | 100 % | 0 % | 100 % | no | |
| fixed: Normal | 94.6 | 100 % | 87 % | 100 % | no | |
| fixed: Aggressive | 72.3 | 100 % | 100 % | 100 % | yes | |
| fixed: Energy saving | -221.2 | 100 % | 0 % | 100 % | no | |
| fixed: Quality focus | 93.6 | 100 % | 100 % | 100 % | yes | |
| fixed: Balanced | 70.0 | 100 % | 46 % | 100 % | no | |
| locked rule | 129.6 | 100 % | 100 % | 100 % | yes | |
| I0 | 44.1 ± 92.4 | 100 % | 37 % | 74 % | 0/6 seeds | 29.6 |
| I1 | 58.9 ± 40.3 | 100 % | 43 % | 94 % | 1/6 seeds | 46.3 |
| I2 | 103.3 ± 10.1 | 100 % | 84 % | 95 % | 3/6 seeds | 105.3 |
| I3 | 98.2 ± 14.1 | 100 % | 72 % | 100 % | 1/6 seeds | 96.3 |
| I4 | 86.3 ± 7.8 | 100 % | 98 % | 88 % | 5/6 seeds | 92.0 |
| I5 | 88.0 ± 8.7 | 100 % | 97 % | 99 % | 5/6 seeds | 93.1 |
| R1 | 88.6 ± 21.0 | 100 % | 70 % | 26 % | 0/6 seeds | 83.8 |

### Preregistered contrasts on held-out nominal (paired by seed; 95 % t-interval, 5 df)

| Contrast | Mean difference (EUR/shift) | 95 % CI | SD between seeds | SD over episodes, within seed | Constraint non-inferiority (S/D/Q) | A feasible | Decision |
|---|---|---|---|---|---|---|---|
| H1: I5 vs I0 | +43.9 | [-53.4, +141.2] | 92.7 | 50.2 | yes/yes/yes | yes | no improvement shown |
| H2: I5 vs locked rule | -41.6 | [-50.3, -32.9] | 8.3 | 11.3 | yes/no/yes | yes | no improvement shown |
| H3: I5 vs best fixed | -5.6 | [-14.3, +3.2] | 8.3 | 7.0 | yes/no/yes | yes | no improvement shown |
| I1 vs I0 | +14.7 | [-83.5, +112.9] | 93.6 | 15.1 | yes/yes/yes | no | no improvement shown |
| I2 vs I0 | +59.1 | [-38.1, +156.3] | 92.6 | 50.7 | yes/yes/yes | no | no improvement shown |
| I3 vs I1 | +39.3 | [-1.4, +80.1] | 38.8 | 45.0 | yes/yes/yes | no | no improvement shown |
| I3 vs I2 | -5.0 | [-19.3, +9.2] | 13.6 | 18.1 | yes/no/yes | no | no improvement shown |
| I4 vs I3 | -11.9 | [-22.2, -1.6] | 9.8 | 25.4 | yes/yes/no | no | no improvement shown |
| I5 vs I4 | +1.7 | [-12.4, +15.8] | 13.4 | 8.5 | yes/yes/yes | yes | no improvement shown |
| R1 vs I3 | -9.6 | [-32.8, +13.5] | 22.1 | 32.1 | yes/yes/no | no | no improvement shown |
| I0 vs best fixed | -49.4 | [-141.8, +43.0] | 88.1 | 52.9 | yes/no/no | no | no improvement shown |
| I0 vs locked rule | -85.5 | [-177.9, +6.9] | 88.1 | 45.5 | yes/no/no | no | no improvement shown |
| I4 vs locked rule | -43.3 | [-51.1, -35.5] | 7.4 | 7.8 | yes/no/no | no | no improvement shown |
| I3 vs locked rule | -31.4 | [-45.5, -17.4] | 13.4 | 20.2 | yes/no/yes | no | no improvement shown |
| interaction (I3-I2)-(I1-I0) | -19.8 | [-119.7, +80.2] | 95.2 | | | | |

### Held-out fault families (100 episodes each): margin and share of episodes failing any grader

| Family | Locked rule | I0 − rule | I5 − rule [95 % CI] | Fail any: rule | I0 | I5 |
|---|---|---|---|---|---|---|
| sensor_bias_low | 72.0 | -64.9 | -85.3 [-173.1, +2.5] | 85 % | 91 % | 94 % |
| sensor_bias_high | 97.9 | -389.6 | -215.0 [-406.7, -23.3] | 100 % | 100 % | 90 % |
| saturation | 120.2 | -136.9 | -84.9 [-110.2, -59.7] | 20 % | 89 % | 91 % |
| delay | 125.1 | -87.1 | -35.9 [-44.4, -27.5] | 5 % | 68 % | 7 % |
| cold_saturation | 115.2 | -150.2 | -125.0 [-142.2, -107.9] | 27 % | 90 % | 100 % |
| bias_lag | 87.0 | -73.1 | -91.0 [-181.3, -0.7] | 76 % | 90 % | 87 % |

Map, I5 − locked rule: mean over the 6 cells inside the development range -40.7 EUR, over the 19 cells outside -37.7 EUR; best cell (cold front 0 K, lag 2) -29.6, worst cell (0 K, lag 1) -42.8. SD of the I5 − rule difference across fault families 60.4 EUR, across map cells 3.7 EUR.

### Compute per run (mean over 6 seeds)

| Condition | Interactions | Gradient updates | Training wall time (s) | CPU time (s) |
|---|---|---|---|---|
| I0 | 240,000 | 1,000 | 156 | 131 |
| I1 | 240,000 | 1,000 | 154 | 114 |
| I2 | 240,000 | 59,985 | 1499 | 511 |
| I3 | 240,000 | 59,985 | 1460 | 483 |
| I4 | 240,000 | 59,985 | 1268 | 486 |
| I5 | 240,000 | 59,985 | 1290 | 505 |
| R1 | 240,000 | 59,985 | 1311 | 507 |

Confirmatory training: 42 runs, 0 failed; wall 48.0 min on 18 worker processes; summed process CPU time 4.56 h. Pilot: wall 4.5 min.
Wall time by stage (s): v0 audit 366, calibration 12, boundary audit 617, pilot 268, confirmatory training 2879, held-out evaluation 105; total 70.8 min.

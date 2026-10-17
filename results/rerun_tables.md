## Table 2: D3IL-Avoiding, SSucc (mean ± std over seeds) / #Vio

| Algorithm | top_left SSucc±std / #Vio | top_right SSucc±std / #Vio | both SSucc±std / #Vio | cluttered SSucc±std / #Vio |
|---|---|---|---|---|
| DPCC | 0.87±n/a / 0.02 | 0.72±n/a / 0.77 | 0.66±n/a / 1.46 | 0.00±n/a / 6.53 |
| SafeDiffuser | 0.21±0.21 / 13.78 [2s, 52t, partial] (safediffuser-res) | 0.05±0.02 / 17.54 [2s, 42t, partial] (safediffuser-res) | 0.03±0.01 / 13.60 [2s, 71t, partial] (safediffuser-tvs) | 0.04±0.04 / 18.68 [2s, 53t, partial] (safediffuser-tvs) |
| MPD | 0.17±0.13 / 7.59 [2s] | 0.14±0.04 / 9.75 [2s] | 0.04±0.01 / 10.53 [2s] | 0.03±0.03 / 10.04 [2s, 140t, partial] |
| Ours (JM2D) | 0.76±n/a / 1.13 | 0.83±n/a / 0.29 | 0.84±n/a / 0.37 | 0.75±n/a / 0.32 [3s] |

_partial cells (`[Ns, Mt, partial]`) include seeds whose 100-trial run was stopped early, aggregated over the trials finished so far (seed<k>.trials.jsonl)._

### Table 2 vs paper

| cell | this | paper | status | source | seeds |
|---|---|---|---|---|---|
| table2.dpcc.top_left.ssucc | 0.8720 | 0.87 | EXACT | stored | 5 |
| table2.dpcc.top_left.ssucc_std | - | 0.04 | NOT STORED | stored | 5 |
| table2.dpcc.top_left.n_vio | 0.0200 | 0.02 | EXACT | stored | 5 |
| table2.dpcc.top_right.ssucc | 0.7180 | 0.72 | EXACT | stored | 5 |
| table2.dpcc.top_right.ssucc_std | - | 0.06 | NOT STORED | stored | 5 |
| table2.dpcc.top_right.n_vio | 0.7740 | 0.77 | EXACT | stored | 5 |
| table2.dpcc.both.ssucc | 0.6640 | 0.66 | EXACT | stored | 5 |
| table2.dpcc.both.ssucc_std | - | 0.15 | NOT STORED | stored | 5 |
| table2.dpcc.both.n_vio | 1.4600 | 1.46 | EXACT | stored | 5 |
| table2.dpcc.cluttered.ssucc | 0.0000 | 0.0 | EXACT | stored | 5 |
| table2.dpcc.cluttered.ssucc_std | - | 0.0 | NOT STORED | stored | 5 |
| table2.dpcc.cluttered.n_vio | 6.5300 | 6.52 | CLOSE | stored | 5 |
| table2.safediffuser.top_left.ssucc | 0.2143 | 0.15 | DIFF | rerun | 2 |
| table2.safediffuser.top_left.ssucc_std | 0.2143 | 0.12 | DIFF | rerun | 2 |
| table2.safediffuser.top_left.n_vio | 13.7768 | 13.18 | DIFF | rerun | 2 |
| table2.safediffuser.top_right.ssucc | 0.0536 | 0.07 | DIFF | rerun | 2 |
| table2.safediffuser.top_right.ssucc_std | 0.0179 | 0.02 | EXACT | rerun | 2 |
| table2.safediffuser.top_right.n_vio | 17.5357 | 18.27 | DIFF | rerun | 2 |
| table2.safediffuser.both.ssucc | 0.0329 | 0.01 | DIFF | rerun | 2 |
| table2.safediffuser.both.ssucc_std | 0.0125 | 0.01 | EXACT | rerun | 2 |
| table2.safediffuser.both.n_vio | 13.5955 | 11.66 | DIFF | rerun | 2 |
| table2.safediffuser.cluttered.ssucc | 0.0417 | 0.01 | DIFF | rerun | 2 |
| table2.safediffuser.cluttered.ssucc_std | 0.0417 | 0.01 | DIFF | rerun | 2 |
| table2.safediffuser.cluttered.n_vio | 18.6806 | 22.92 | DIFF | rerun | 2 |
| table2.mpd.top_left.ssucc | 0.1700 | 0.13 | DIFF | rerun | 2 |
| table2.mpd.top_left.ssucc_std | 0.1300 | 0.13 | EXACT | rerun | 2 |
| table2.mpd.top_left.n_vio | 7.5900 | 11.97 | DIFF | rerun | 2 |
| table2.mpd.top_right.ssucc | 0.1350 | 0.08 | DIFF | rerun | 2 |
| table2.mpd.top_right.ssucc_std | 0.0450 | 0.06 | DIFF | rerun | 2 |
| table2.mpd.top_right.n_vio | 9.7550 | 14.85 | DIFF | rerun | 2 |
| table2.mpd.both.ssucc | 0.0450 | 0.03 | DIFF | rerun | 2 |
| table2.mpd.both.ssucc_std | 0.0150 | 0.03 | DIFF | rerun | 2 |
| table2.mpd.both.n_vio | 10.5250 | 10.8 | DIFF | rerun | 2 |
| table2.mpd.cluttered.ssucc | 0.0250 | 0.02 | CLOSE | rerun | 2 |
| table2.mpd.cluttered.ssucc_std | 0.0250 | 0.02 | CLOSE | rerun | 2 |
| table2.mpd.cluttered.n_vio | 10.0425 | 12.71 | DIFF | rerun | 2 |
| table2.jm2d.top_left.ssucc | 0.7640 | 0.77 | CLOSE | stored | 5 |
| table2.jm2d.top_left.ssucc_std | - | 0.1 | NOT STORED | stored | 5 |
| table2.jm2d.top_left.n_vio | 1.1260 | 1.13 | EXACT | stored | 5 |
| table2.jm2d.top_right.ssucc | 0.8320 | 0.83 | EXACT | stored | 5 |
| table2.jm2d.top_right.ssucc_std | - | 0.15 | NOT STORED | stored | 5 |
| table2.jm2d.top_right.n_vio | 0.2880 | 0.29 | EXACT | stored | 5 |
| table2.jm2d.both.ssucc | 0.8360 | 0.84 | EXACT | stored | 5 |
| table2.jm2d.both.ssucc_std | - | 0.1 | NOT STORED | stored | 5 |
| table2.jm2d.both.n_vio | 0.3680 | 0.37 | EXACT | stored | 5 |
| table2.jm2d.cluttered.ssucc | 0.7500 | 0.75 | EXACT | stored | 3 |
| table2.jm2d.cluttered.ssucc_std | - | 0.01 | NOT STORED | stored | 3 |
| table2.jm2d.cluttered.n_vio | 0.3233 | 0.32 | EXACT | stored | 3 |

tally: CLOSE 4, DIFF 19, EXACT 17, NOT STORED 8

### Reproduction checks: DPCC and Ours, re-run vs stored vs paper

re-run: mean ± std [min, max] over seeds (number of seeds); stored: mean [min, max] (std not stored); paper: mean ± std. Cells: SSucc / #Vio.

| row | source | top_left SSucc / #Vio | top_right SSucc / #Vio | both SSucc / #Vio | cluttered SSucc / #Vio |
|---|---|---|---|---|---|
| DPCC | rerun | 0.880±0.05 [0.80, 0.95] / 0.000 (5s) | 0.680±0.10 [0.50, 0.75] / 0.710 (5s) | 0.730±0.09 [0.60, 0.85] / 1.430 (5s) | 0.000±0.00 [0.00, 0.00] / 8.040 (5s) |
| DPCC | stored | 0.872 [0.81, 0.92] / 0.020 | 0.718 [0.65, 0.80] / 0.774 | 0.664 [0.49, 0.81] / 1.460 | 0.000 [0.00, 0.00] / 6.530 |
| DPCC | paper | 0.87±0.04 / 0.02 | 0.72±0.06 / 0.77 | 0.66±0.15 / 1.46 | 0.00±0.00 / 6.52 |
| Ours (JM2D) | rerun | 0.820±0.08 [0.70, 0.90] / 0.930 (5s) | 0.860±0.16 [0.55, 1.00] / 0.030 (5s) | 0.820±0.13 [0.65, 1.00] / 0.320 (5s) | 0.780±0.16 [0.55, 1.00] / 0.220 (5s) |
| Ours (JM2D) | stored | 0.764 [0.64, 0.87] / 1.126 | 0.832 [0.55, 0.98] / 0.288 | 0.836 [0.63, 0.94] / 0.368 | 0.750 [0.74, 0.76] / 0.323 (3s) |
| Ours (JM2D) | paper | 0.77±0.10 / 1.13 | 0.83±0.15 / 0.29 | 0.84±0.10 / 0.37 | 0.75±0.01 / 0.32 |

### Re-run DPCC / Ours vs paper

| cell | this | paper | status | source | seeds |
|---|---|---|---|---|---|
| table2_rerun.dpcc.top_left.ssucc | 0.8800 | 0.87 | CLOSE | rerun | 5 |
| table2_rerun.dpcc.top_left.ssucc_std | 0.0510 | 0.04 | DIFF | rerun | 5 |
| table2_rerun.dpcc.top_left.n_vio | 0.0000 | 0.02 | DIFF | rerun | 5 |
| table2_rerun.dpcc.top_right.ssucc | 0.6800 | 0.72 | DIFF | rerun | 5 |
| table2_rerun.dpcc.top_right.ssucc_std | 0.0980 | 0.06 | DIFF | rerun | 5 |
| table2_rerun.dpcc.top_right.n_vio | 0.7100 | 0.77 | DIFF | rerun | 5 |
| table2_rerun.dpcc.both.ssucc | 0.7300 | 0.66 | DIFF | rerun | 5 |
| table2_rerun.dpcc.both.ssucc_std | 0.0927 | 0.15 | DIFF | rerun | 5 |
| table2_rerun.dpcc.both.n_vio | 1.4300 | 1.46 | DIFF | rerun | 5 |
| table2_rerun.dpcc.cluttered.ssucc | 0.0000 | 0.0 | EXACT | rerun | 5 |
| table2_rerun.dpcc.cluttered.ssucc_std | 0.0000 | 0.0 | EXACT | rerun | 5 |
| table2_rerun.dpcc.cluttered.n_vio | 8.0400 | 6.52 | DIFF | rerun | 5 |
| table2_rerun.jm2d.top_left.ssucc | 0.8200 | 0.77 | DIFF | rerun | 5 |
| table2_rerun.jm2d.top_left.ssucc_std | 0.0812 | 0.1 | DIFF | rerun | 5 |
| table2_rerun.jm2d.top_left.n_vio | 0.9300 | 1.13 | DIFF | rerun | 5 |
| table2_rerun.jm2d.top_right.ssucc | 0.8600 | 0.83 | DIFF | rerun | 5 |
| table2_rerun.jm2d.top_right.ssucc_std | 0.1594 | 0.15 | CLOSE | rerun | 5 |
| table2_rerun.jm2d.top_right.n_vio | 0.0300 | 0.29 | DIFF | rerun | 5 |
| table2_rerun.jm2d.both.ssucc | 0.8200 | 0.84 | DIFF | rerun | 5 |
| table2_rerun.jm2d.both.ssucc_std | 0.1327 | 0.1 | DIFF | rerun | 5 |
| table2_rerun.jm2d.both.n_vio | 0.3200 | 0.37 | DIFF | rerun | 5 |
| table2_rerun.jm2d.cluttered.ssucc | 0.7800 | 0.75 | DIFF | rerun | 5 |
| table2_rerun.jm2d.cluttered.ssucc_std | 0.1631 | 0.01 | DIFF | rerun | 5 |
| table2_rerun.jm2d.cluttered.n_vio | 0.2200 | 0.32 | DIFF | rerun | 5 |

tally: CLOSE 2, DIFF 20, EXACT 2

## Table 3: JM2D u-step ablation (--u-steps u), SSucc (mean ± std) / #Vio

| u | top_left SSucc±std / #Vio | top_right SSucc±std / #Vio | both SSucc±std / #Vio | cluttered SSucc±std / #Vio |
|---|---|---|---|---|
| u = 0 | 0.15±0.16 / 12.82 | 0.05±0.05 / 18.73 | 0.01±0.02 / 12.32 | 0.02±0.02 / 22.85 |
| u = 1 | 0.39±0.23 / 3.52 | 0.27±0.16 / 6.55 | 0.33±0.11 / 5.33 | 0.57±0.15 / 1.30 |
| u = 2 | 0.41±0.27 / 3.86 | 0.38±0.17 / 3.20 | 0.51±0.17 / 3.85 | 0.54±0.17 / 1.04 |
| u = 5 | 0.67±0.12 / 1.98 | 0.63±0.19 / 0.75 | 0.80±0.09 / 0.47 | 0.67±0.14 / 0.61 |
| u = 10 | 0.77±0.16 / 1.20 | 0.77±0.12 / 0.81 | 0.85±0.10 / 0.54 | 0.73±0.11 / 0.44 |

_partial cells (`[Ns, Mt, partial]`) include seeds whose 100-trial run was stopped early, aggregated over the trials finished so far (seed<k>.trials.jsonl)._

### Table 3 vs paper

| cell | this | paper | status | source | seeds |
|---|---|---|---|---|---|
| table3_u_steps.0.top_left.ssucc | 0.1500 | 0.13 | DIFF | rerun | 5 |
| table3_u_steps.0.top_left.ssucc_std | 0.1581 | 0.15 | CLOSE | rerun | 5 |
| table3_u_steps.0.top_left.n_vio | 12.8200 | 12.92 | DIFF | rerun | 5 |
| table3_u_steps.0.top_right.ssucc | 0.0500 | 0.06 | CLOSE | rerun | 5 |
| table3_u_steps.0.top_right.ssucc_std | 0.0548 | 0.07 | DIFF | rerun | 5 |
| table3_u_steps.0.top_right.n_vio | 18.7300 | 18.34 | DIFF | rerun | 5 |
| table3_u_steps.0.both.ssucc | 0.0100 | 0.0 | CLOSE | rerun | 5 |
| table3_u_steps.0.both.ssucc_std | 0.0200 | 0.02 | EXACT | rerun | 5 |
| table3_u_steps.0.both.n_vio | 12.3200 | 12.98 | DIFF | rerun | 5 |
| table3_u_steps.0.cluttered.ssucc | 0.0200 | 0.01 | CLOSE | rerun | 5 |
| table3_u_steps.0.cluttered.ssucc_std | 0.0245 | 0.02 | EXACT | rerun | 5 |
| table3_u_steps.0.cluttered.n_vio | 22.8500 | 23.19 | DIFF | rerun | 5 |
| table3_u_steps.1.top_left.ssucc | 0.3900 | 0.47 | DIFF | rerun | 5 |
| table3_u_steps.1.top_left.ssucc_std | 0.2289 | 0.24 | DIFF | rerun | 5 |
| table3_u_steps.1.top_left.n_vio | 3.5200 | 3.54 | DIFF | rerun | 5 |
| table3_u_steps.1.top_right.ssucc | 0.2700 | 0.33 | DIFF | rerun | 5 |
| table3_u_steps.1.top_right.ssucc_std | 0.1568 | 0.23 | DIFF | rerun | 5 |
| table3_u_steps.1.top_right.n_vio | 6.5500 | 5.24 | DIFF | rerun | 5 |
| table3_u_steps.1.both.ssucc | 0.3300 | 0.32 | CLOSE | rerun | 5 |
| table3_u_steps.1.both.ssucc_std | 0.1077 | 0.18 | DIFF | rerun | 5 |
| table3_u_steps.1.both.n_vio | 5.3300 | 5.76 | DIFF | rerun | 5 |
| table3_u_steps.1.cluttered.ssucc | 0.5700 | 0.34 | DIFF | rerun | 5 |
| table3_u_steps.1.cluttered.ssucc_std | 0.1503 | 0.12 | DIFF | rerun | 5 |
| table3_u_steps.1.cluttered.n_vio | 1.3000 | 1.35 | DIFF | rerun | 5 |
| table3_u_steps.2.top_left.ssucc | 0.4100 | 0.5 | DIFF | rerun | 5 |
| table3_u_steps.2.top_left.ssucc_std | 0.2672 | 0.3 | DIFF | rerun | 5 |
| table3_u_steps.2.top_left.n_vio | 3.8600 | 3.14 | DIFF | rerun | 5 |
| table3_u_steps.2.top_right.ssucc | 0.3800 | 0.45 | DIFF | rerun | 5 |
| table3_u_steps.2.top_right.ssucc_std | 0.1749 | 0.17 | EXACT | rerun | 5 |
| table3_u_steps.2.top_right.n_vio | 3.2000 | 2.8 | DIFF | rerun | 5 |
| table3_u_steps.2.both.ssucc | 0.5100 | 0.5 | CLOSE | rerun | 5 |
| table3_u_steps.2.both.ssucc_std | 0.1744 | 0.16 | DIFF | rerun | 5 |
| table3_u_steps.2.both.n_vio | 3.8500 | 3.48 | DIFF | rerun | 5 |
| table3_u_steps.2.cluttered.ssucc | 0.5400 | 0.39 | DIFF | rerun | 5 |
| table3_u_steps.2.cluttered.ssucc_std | 0.1715 | 0.15 | DIFF | rerun | 5 |
| table3_u_steps.2.cluttered.n_vio | 1.0400 | 1.11 | DIFF | rerun | 5 |
| table3_u_steps.5.top_left.ssucc | 0.6700 | 0.68 | CLOSE | rerun | 5 |
| table3_u_steps.5.top_left.ssucc_std | 0.1208 | 0.23 | DIFF | rerun | 5 |
| table3_u_steps.5.top_left.n_vio | 1.9800 | 1.83 | DIFF | rerun | 5 |
| table3_u_steps.5.top_right.ssucc | 0.6300 | 0.65 | DIFF | rerun | 5 |
| table3_u_steps.5.top_right.ssucc_std | 0.1860 | 0.19 | EXACT | rerun | 5 |
| table3_u_steps.5.top_right.n_vio | 0.7500 | 1.15 | DIFF | rerun | 5 |
| table3_u_steps.5.both.ssucc | 0.8000 | 0.82 | DIFF | rerun | 5 |
| table3_u_steps.5.both.ssucc_std | 0.0894 | 0.09 | EXACT | rerun | 5 |
| table3_u_steps.5.both.n_vio | 0.4700 | 0.59 | DIFF | rerun | 5 |
| table3_u_steps.5.cluttered.ssucc | 0.6700 | 0.63 | DIFF | rerun | 5 |
| table3_u_steps.5.cluttered.ssucc_std | 0.1435 | 0.12 | DIFF | rerun | 5 |
| table3_u_steps.5.cluttered.n_vio | 0.6100 | 0.65 | DIFF | rerun | 5 |
| table3_u_steps.10.top_left.ssucc | 0.7700 | 0.7 | DIFF | rerun | 5 |
| table3_u_steps.10.top_left.ssucc_std | 0.1600 | 0.22 | DIFF | rerun | 5 |
| table3_u_steps.10.top_left.n_vio | 1.2000 | 1.6 | DIFF | rerun | 5 |
| table3_u_steps.10.top_right.ssucc | 0.7700 | 0.81 | DIFF | rerun | 5 |
| table3_u_steps.10.top_right.ssucc_std | 0.1208 | 0.19 | DIFF | rerun | 5 |
| table3_u_steps.10.top_right.n_vio | 0.8100 | 0.41 | DIFF | rerun | 5 |
| table3_u_steps.10.both.ssucc | 0.8500 | 0.84 | CLOSE | rerun | 5 |
| table3_u_steps.10.both.ssucc_std | 0.1000 | 0.07 | DIFF | rerun | 5 |
| table3_u_steps.10.both.n_vio | 0.5400 | 0.38 | DIFF | rerun | 5 |
| table3_u_steps.10.cluttered.ssucc | 0.7300 | 0.73 | EXACT | rerun | 5 |
| table3_u_steps.10.cluttered.ssucc_std | 0.1077 | 0.09 | DIFF | rerun | 5 |
| table3_u_steps.10.cluttered.n_vio | 0.4400 | 0.49 | DIFF | rerun | 5 |

tally: CLOSE 8, DIFF 46, EXACT 6


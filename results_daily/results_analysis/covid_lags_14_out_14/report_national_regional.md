# Análise COVID-19 Brasil — lags_14_out_14

## Eventos

|   event |   snapshot |   peak_total_cases |   pre_peak_total |   absolute_rise |   relative_rise | date                |
|--------:|-----------:|-------------------:|-----------------:|----------------:|----------------:|:--------------------|
|       1 |         99 |            5892.43 |          4193.86 |         1698.57 |        0.405014 | 2022-10-26 00:00:00 |
|       2 |        146 |           47510.3  |         22871.6  |        24638.7  |        1.07726  | 2022-12-13 00:00:00 |

## Métricas

| model       | region       | phase   |      RMSE |       MAE |
|:------------|:-------------|:--------|----------:|----------:|
| GCLSTM      | Brasil       | decline | 17405.7   | 15018.5   |
| GCLSTM      | Brasil       | peak    | 14888.6   | 13162.2   |
| GCLSTM      | Brasil       | surge   | 15279.7   | 13131.6   |
| GCLSTM      | Centro-Oeste | decline |  2153.14  |  1744.66  |
| GCLSTM      | Centro-Oeste | peak    |  3312.68  |  2576.15  |
| GCLSTM      | Centro-Oeste | surge   |  2205.42  |  1695.38  |
| GCLSTM      | Nordeste     | decline |  4974.35  |  4566.21  |
| GCLSTM      | Nordeste     | peak    |  5231.32  |  4811.32  |
| GCLSTM      | Nordeste     | surge   |  4854.12  |  4217.12  |
| GCLSTM      | Norte        | decline |  1244.07  |  1127.56  |
| GCLSTM      | Norte        | peak    |  1283.78  |  1164.21  |
| GCLSTM      | Norte        | surge   |  1011.99  |   893.733 |
| GCLSTM      | Sudeste      | decline |  8025.43  |  6632.8   |
| GCLSTM      | Sudeste      | peak    |  6919.09  |  5640.84  |
| GCLSTM      | Sudeste      | surge   |  6844.9   |  5853.91  |
| GCLSTM      | Sul          | decline |  3682.4   |  3241.95  |
| GCLSTM      | Sul          | peak    |  2830.88  |  2448     |
| GCLSTM      | Sul          | surge   |  3396.86  |  2866.06  |
| GCRN        | Brasil       | decline | 16926.6   | 14445.3   |
| GCRN        | Brasil       | peak    | 14442.8   | 12682.4   |
| GCRN        | Brasil       | surge   | 14867.9   | 12602.5   |
| GCRN        | Centro-Oeste | decline |  2151.96  |  1746.08  |
| GCRN        | Centro-Oeste | peak    |  3319.71  |  2582.42  |
| GCRN        | Centro-Oeste | surge   |  2204.3   |  1693.74  |
| GCRN        | Nordeste     | decline |  4342.18  |  3856.4   |
| GCRN        | Nordeste     | peak    |  4623.32  |  4118.78  |
| GCRN        | Nordeste     | surge   |  4438.35  |  3770.81  |
| GCRN        | Norte        | decline |  1190.02  |  1063.52  |
| GCRN        | Norte        | peak    |  1232.43  |  1097.51  |
| GCRN        | Norte        | surge   |   950.459 |   827.659 |
| GCRN        | Sudeste      | decline |  8028.94  |  6621.72  |
| GCRN        | Sudeste      | peak    |  6904.34  |  5625.54  |
| GCRN        | Sudeste      | surge   |  6802.7   |  5809.23  |
| GCRN        | Sul          | decline |  3692.47  |  3245.06  |
| GCRN        | Sul          | peak    |  2825.79  |  2438.64  |
| GCRN        | Sul          | surge   |  3385.43  |  2853.57  |
| LSTM        | Brasil       | decline | 42648.4   | 40957.4   |
| LSTM        | Brasil       | peak    | 37875.5   | 35922.7   |
| LSTM        | Brasil       | surge   | 37258.5   | 34968.2   |
| LSTM        | Centro-Oeste | decline |  3213.77  |  2773.36  |
| LSTM        | Centro-Oeste | peak    |  3844.53  |  3260.87  |
| LSTM        | Centro-Oeste | surge   |  2937.46  |  2557.34  |
| LSTM        | Nordeste     | decline | 20620     | 20483.2   |
| LSTM        | Nordeste     | peak    | 20462.9   | 20338.5   |
| LSTM        | Nordeste     | surge   | 19171.8   | 18812.1   |
| LSTM        | Norte        | decline |  3491.47  |  3415.26  |
| LSTM        | Norte        | peak    |  3421.54  |  3313.24  |
| LSTM        | Norte        | surge   |  3327.35  |  3254.23  |
| LSTM        | Sudeste      | decline | 11803.4   | 10591.7   |
| LSTM        | Sudeste      | peak    | 11012.1   |  9840.93  |
| LSTM        | Sudeste      | surge   | 11049.9   |  9964.56  |
| LSTM        | Sul          | decline |  5684.08  |  5220.4   |
| LSTM        | Sul          | peak    |  4468.55  |  3983.4   |
| LSTM        | Sul          | surge   |  4813.11  |  4352.92  |
| Persistence | Brasil       | decline | 16112     | 13364.2   |
| Persistence | Brasil       | peak    | 14402.2   | 12356.9   |
| Persistence | Brasil       | surge   | 14808.9   | 12401     |
| Persistence | Centro-Oeste | decline |  2260.5   |  1935.95  |
| Persistence | Centro-Oeste | peak    |  3889.25  |  3276.3   |
| Persistence | Centro-Oeste | surge   |  2466.51  |  1852.43  |
| Persistence | Nordeste     | decline |  2770.56  |  2314.41  |
| Persistence | Nordeste     | peak    |  2535.65  |  2170.25  |
| Persistence | Nordeste     | surge   |  3572.78  |  2907.77  |
| Persistence | Norte        | decline |   937.148 |   730.768 |
| Persistence | Norte        | peak    |  1021.85  |   810.883 |
| Persistence | Norte        | surge   |   702.719 |   581.809 |
| Persistence | Sudeste      | decline |  8856.08  |  7224.33  |
| Persistence | Sudeste      | peak    |  8372.6   |  6971.55  |
| Persistence | Sudeste      | surge   |  7894.89  |  6581.73  |
| Persistence | Sul          | decline |  3668.87  |  2990.81  |
| Persistence | Sul          | peak    |  3503.97  |  2963.71  |
| Persistence | Sul          | surge   |  3523.68  |  2817.35  |

## Conformal Prediction

| model       | phase   |   coverage_95 |   interval_width |   quantile_95 |
|:------------|:--------|--------------:|-----------------:|--------------:|
| GCLSTM      | decline |      0.932399 |          28.4564 |       14.2282 |
| GCLSTM      | peak    |      0.92393  |          28.4564 |       14.2282 |
| GCLSTM      | surge   |      0.952626 |          28.4564 |       14.2282 |
| GCRN        | decline |      0.93248  |          28.6238 |       14.3119 |
| GCRN        | peak    |      0.924052 |          28.6238 |       14.3119 |
| GCRN        | surge   |      0.952628 |          28.6238 |       14.3119 |
| LSTM        | decline |      0.919023 |          34.6672 |       17.3336 |
| LSTM        | peak    |      0.922344 |          34.6672 |       17.3336 |
| LSTM        | surge   |      0.955522 |          34.6672 |       17.3336 |
| Persistence | decline |      0.940578 |          36.5135 |       18.2567 |
| Persistence | peak    |      0.922765 |          36.5135 |       18.2567 |
| Persistence | surge   |      0.951529 |          36.5135 |       18.2567 |

## Cidades importantes

|   ibgeID | city           |   n_metrics_in_top |   mean_value |   rank |
|---------:|:---------------|-------------------:|-------------:|-------:|
|  3550308 | São Paulo      |                  3 |    681472    |      1 |
|  3304557 | Rio de Janeiro |                  2 |    233934    |      2 |
|  2211001 | Teresina       |                  2 |    178127    |      3 |
|  2927408 | Salvador       |                  2 |     12507.2  |      4 |
|  5300108 | Brasília       |                  2 |      9277.12 |      5 |

# Montreal Local TV Channels: Market Share Prediction Task

The goal of this task is to build a model using a supervised approach to make a relationship between market share amount and some features for predicting how much share new channels in Monreal local tv will take.

I proposed the Ensemble Learning approach called `BaggingRegressor` which works quite well on capturing the train data, with the lowest `mean_absolute_error` and the highest `R2 Squared` score. My results with performing 5-fold cross-validation are in the bellow table.

| 5-fold Cross-Validation (mean +/- std) | R2 Squared | MAE|
|:---:|:---:|:---:|
| 0.75178  +/-  0.0403 |  0.97424 |0.42643|

and to perform my ensemble approach:

```
python main.py --train data.csv --test test.csv
```


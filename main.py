import argparse
import Read_and_Write_Data as rw
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', help="path to train set")
    parser.add_argument('-ts', '--test' , help="path to test set")
    args = parser.parse_args()
    if args.train is None or  args.test is None:
        parser.print_usage()
        exit()
    return args


def main():
    arg = args()
    train = rw.read(arg.train)
    test = rw.read(arg.test)
    X = train.loc[:, train.columns != 'Market Share_total']
    y = train['Market Share_total']
    bagging_regressor = BaggingRegressor()
    bagging_regressor.fit(X, y)
    predictions = bagging_regressor.predict(test)
    rw.write(predictions, "test_results.csv")

if __name__ == "__main__":
    main()
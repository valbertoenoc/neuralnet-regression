"""Train neural net and generate prediction file results.
Usage:
    neuralnet-train-predict.py <train_data.csv> <ground_truth.csv> <test_data.csv>
"""
import numpy as np
import pandas as pd
from docopt import docopt

from sklearn.neural_network import MLPRegressor

def trainModel(x_train, y_train):
    model = MLPRegressor(solver='adam', hidden_layer_sizes=(50,23,23),
                             max_iter=500, shuffle=True, random_state=1, alpha=0.01,
                             activation="relu", verbose=False)
    model.fit(x_train, y_train)

    return model

def testModel(model, x_test):
    predictions = model.predict(x_test)

    return predictions

def main(args):
    train_fn = args['<train_data.csv>']
    truth_fn = args['<ground_truth.csv>']
    test_fn = args['<test_data.csv>']

    train_data = pd.read_csv(train_fn)
    ground_truth = pd.read_csv(truth_fn)
    test_data = pd.read_csv(test_fn)

    x_train = train_data.drop(['id'], axis=1)
    y_train = ground_truth.drop(['id'], axis=1)
    x_test = test_data.drop(['id'], axis=1)

    model = trainModel(x_train, y_train)
    prediction = testModel(model, x_test)

    # create submission data frame
    submission_df = pd.DataFrame({
        'id': ground_truth['id'],
        'slope': prediction[:,0],
        'intercept': prediction[:,1]
    })

    # write output to file
    submission_df.to_csv('submission.csv', index=False, columns=['id', 'slope', 'intercept'])
    

if __name__ == '__main__':
    main(docopt(__doc__))
import argparse
import time
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from model import mlModel

def preProcess(dataFile):
    df = pandas.read_csv(dataFile)
    print(df.head())

def linearModel(dataFile):
    df = pandas.read_csv(dataFile)
    print(df.head())
    dataSet = {}
    count = 0
    df2 = df.iloc[:, -1].copy()
    for count1, label in enumerate(df2):
        if label in dataSet:
            df2.loc[count1] = dataSet[label]
        else:
            dataSet[label] = count
            df2.loc[count1] = count
            count+=1
    print(f'String to number mapping: {dataSet}')
    X = df.iloc[:,:-1]
    y = df2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f'Accuracy: {model.score(X_test, y_test)}')


if __name__ == '__main__':
    startTime = time.time()
    parser = argparse.ArgumentParser(
        description='CLI for Machine Learning Model generation')
    parser.add_argument('data', help='csv file containing data')
    parser.add_argument('modelType', choices=['linear', 'logistic'], default='linear',nargs='?', help='Model Type')
    # parser.add_argument()
    parser.add_argument('-s', '--string', help='set string labels', action='store_true')
    args = parser.parse_args()
    # linearModel(args.data)
    model = mlModel(args.data, args.modelType, args.string, .33)
    print(f"Runtime: {round(time.time() - startTime, 2)} seconds")
    # test




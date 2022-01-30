import argparse
import time
from model import mlModel


def main():
    startTime = time.time()
    x = ''
    while x.lower() != 'y' and x.lower() != 'yes':
        parser = argparse.ArgumentParser(
            description='CLI for Machine Learning Model generation')
        parser.add_argument('data', help='csv file containing data')
        parser.add_argument('modelType', choices=['linear', 'logistic', 'svm'], default='linear',nargs='?', help='Model Type')
        parser.add_argument('-p', '--plot', help='set plot to show', action='store_true')
        parser.add_argument('-s', '--string', help='set string labels', action='store_true')
        parser.add_argument('-c', '--column', help='choose label column (default is last column)', default=0)
        parser.add_argument('-t', '--test', help='choose the size of test set (default=.33)', type=float, default=.33)

        args = parser.parse_args()
        print(args.column)
        model = mlModel(args.data, args.modelType, args.string, args.plot, args.column, args.test)
        print(f"Model Runtime: {round(time.time() - startTime, 2)} seconds")

        x = input('Exit?: ')

    print(f"Script Runtime: {round(time.time() - startTime, 2)} seconds")

if __name__ == '__main__':
    main()




# mlcli
 
Command line application for fast machine learning model generation.


usage: main.py [-h] [-p] [-s] [-c COLUMN] [-t TEST] [--predict] data [{linear,logistic,svm}]

CLI for Machine Learning Model generation

positional arguments:
  data                  csv file containing data
  {linear,logistic,svm}
                        Model Type

optional arguments:
  -h, --help            show this help message and exit
  -p, --plot            set plot to show
  -s, --string          set string labels
  -c COLUMN, --column COLUMN
                        choose label column (default is last column)
  -t TEST, --test TEST  choose the size of test set (default=.33)
  --predict             allow prediction of new sample with model

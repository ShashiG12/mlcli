import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import time


class mlModel:

    # modelType - type of model [linear]
    # stringLabels - whether labels are strings
    # testPercentage - percent of data to be used for testing
    def __init__(self, dataFile, modelType, stringLabels, showPlot, labelColumn, testPercentage, isPredict):
        startTime = time.time()
        self.dataFile = dataFile
        self.df = pandas.read_csv(self.dataFile)
        self.modelType = modelType
        self.stringLabels = stringLabels
        self.showPlot = showPlot
        self.isPredict = isPredict

        self.model = None

        # check for default last column of features
        if labelColumn == 0:
            self.labelColumn = len(self.df.columns)
        else:
            self.labelColumn = labelColumn
        # self.labelColumn = labelColumn

        self.testPercentage = testPercentage
        self.X_train, self.X_test, self.y_train, self.y_test = 0,0,0,0

        print(self.df.head())

        # check if the label column is made of strings
        if stringLabels:
            self.stringLabelToNum()
        else:
            self.normalLabels()
        self.play()
        print(f"Model Runtime: {round(time.time() - startTime, 2)} seconds")
        self.predict()

    # convert string label column to nums and return corresponding dictionary
    def stringLabelToNum(self):
        # converts any number of strings labels to nums using a dictionary
        dataSet = {}
        count = 0
        df2 = self.df.iloc[:, -1].copy()
        for count1, label in enumerate(df2):
            if label in dataSet:
                df2.loc[count1] = dataSet[label]
            else:
                dataSet[label] = count
                df2.loc[count1] = count
                count += 1
        print(f'String to number mapping: {dataSet}')
        X = self.df.iloc[:, :-1]
        y = df2
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.testPercentage)

    def normalLabels(self):
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1].copy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.testPercentage)

    # run correct ML model
    def play(self):
        if self.modelType == 'linear':
            self.linearModel()
        if self.modelType == 'logistic':
            self.logisticModel()
        if self.modelType == 'svm':
            self.svmModel()

    def linearModel(self):
        self.model = LinearRegression()
        self.afterModel()

    def logisticModel(self):
        self.model = LogisticRegression(multi_class='multinomial')
        self.afterModel()

    def svmModel(self):
        self.model = SVC(gamma='auto')
        self.afterModel()

    def afterModel(self):
        self.model.fit(self.X_train, self.y_train)
        print(f'Accuracy: {self.model.score(self.X_test, self.y_test)}')
        self.plot()

    # scatter plot the data with predictions
    def plot(self):
        if self.showPlot:
            y_pred = self.model.predict(self.X_test)
            plt.scatter(self.X_test.iloc[:,0], self.y_test)
            plt.plot(self.X_test.iloc[:,0], y_pred)
            plt.show()

    def predict(self):
        if self.isPredict:
            y = [[]]
            # x = ''
            # while x.lower() != 'q' and x.lower() != 'quit':
            #     for column in self.df.columns():
            #         x = input('Input: ')
            #         self.model.predict(x)
            for i, column in enumerate(self.df.columns):
                y[0].append(float(input(f'{self.df.columns[i]}: ')))
            y[0].pop(4)
            print(self.model.predict(y)[0])

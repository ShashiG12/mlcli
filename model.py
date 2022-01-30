import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class mlModel:

    # modelType - type of model [linear]
    # stringLabels - whether labels are strings
    # testPercentage - percent of data to be used for testing
    def __init__(self, dataFile, modelType, stringLabels, showPlot, labelColumn, testPercentage):
        self.dataFile = dataFile
        self.df = pandas.read_csv(self.dataFile)
        self.modelType = modelType
        self.stringLabels = stringLabels
        self.showPlot = showPlot

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

    # convert string label column to nums and return corresponding dictionary
    def stringLabelToNum(self):
        # df = pandas.read_csv(self.dataFile)
        # print(df.head())

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
        # df = pandas.read_csv(self.dataFile)
        # print(df.head())

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
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        print(f'Accuracy: {model.score(self.X_test, self.y_test)}')
        self.plot(model)

    def logisticModel(self):
        model = LogisticRegression(multi_class='multinomial')
        model.fit(self.X_train, self.y_train)
        print(f'Accuracy: {model.score(self.X_test, self.y_test)}')
        self.plot(model)

    def svmModel(self):
        model = SVC(gamma='auto')
        model.fit(self.X_train, self.y_train)
        print(f'Accuracy: {model.score(self.X_test, self.y_test)}')
        self.plot(model)

    # scatter plot the data with predictions
    def plot(self, model):
        if self.showPlot:
            y_pred = model.predict(self.X_test)
            plt.scatter(self.X_test.iloc[:,0], self.y_test)
            plt.plot(self.X_test.iloc[:,0], y_pred)
            plt.show()
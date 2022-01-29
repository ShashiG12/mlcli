import sklearn
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


class mlModel:

    # modelType - type of model [linear]
    # stringLabels - whether labels are strings
    # testPercentage - percent of data to be used for testing
    def __init__(self, dataFile, modelType, stringLabels, testPercentage):
        self.dataFile = dataFile
        self.modelType = modelType
        self.stringLabels = stringLabels
        self.testPercentage = testPercentage

        df = pandas.read_csv(self.dataFile)
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
                count += 1
        print(f'String to number mapping: {dataSet}')
        X = df.iloc[:, :-1]
        y = df2
        self. X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.testPercentage)
        self.play()

    def play(self):
        if self.modelType == 'linear':
            self.linearModel()
        if self.modelType == 'logistic':
            self.logisticModel()

    def linearModel(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        print(f'Accuracy: {model.score(self.X_test, self.y_test)}')

    def logisticModel(self):
        model = LogisticRegression(multi_class='multinomial')
        model.fit(self.X_train, self.y_train)
        print(f'Accuracy: {model.score(self.X_test, self.y_test)}')
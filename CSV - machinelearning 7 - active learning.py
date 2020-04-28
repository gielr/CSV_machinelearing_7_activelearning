import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

Encode = preprocessing.LabelEncoder()

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

import xlsxwriter


# split dataset into test set, train set and unlabel pool
def split(dataset, train_size, test_size):
    x = dataset.iloc[:, 0]
    y = dataset.iloc[:, 1]
    x_train, x_pool, y_train, y_pool = train_test_split(
        x, y, train_size=train_size)
    unlabel, x_test, label, y_test = train_test_split(
        x_pool, y_pool, test_size=test_size)
    return x_train, y_train, x_test, y_test, unlabel, label


def splitValidation(datasetValidation):
    x_validation = datasetValidation.iloc[:, 0]
    y_validation = datasetValidation.iloc[:, 1]
    return x_validation, y_validation


def adjust(opis_train, kategoria_train, opis_test, kategoria_test):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', clf), ])
    text_clf.fit(opis_train, kategoria_train)

    predicted = text_clf.predict(opis_test)
    accuracy = metrics.accuracy_score(kategoria_test, predicted)
    return accuracy


if __name__ == '__main__':
    dataset = pd.read_csv('Files\Dane-do-nauki.csv', delimiter=';', encoding='utf-8')
    datasetValidation = pd.read_csv('Files/Dane-do-walidacji.csv', delimiter=';', encoding='utf-8')
    x_validation, y_validation = splitValidation(datasetValidation)

    # Tworze plik z wynikami
    results = xlsxwriter.Workbook("Wyniki.xlsx")
    worksheet = results.add_worksheet("Wyniki")
    row = 0
    col = 0

    # Zapisuje nazwy kolumn
    worksheet.write(row, col, "Nazwa_algorytmu")
    worksheet.write(row, col + 1, "Acc")
    row += 1

    names = ["RandomForestClassifier", "KNeighborsClassifier", "MultinomialNB", "DecisionTreeClassifier", "BernoulliNB",
             "AdaBoostClassifier", "LogisticRegression", "SVC1", "SVC2", "SVC3"]
    classifiers = [
        RandomForestClassifier(max_depth=10000, n_estimators=100, max_features=100),
        KNeighborsClassifier(3),
        MultinomialNB(),
        DecisionTreeClassifier(max_depth=10000),
        BernoulliNB(),
        AdaBoostClassifier(),
        LogisticRegression(),
        SVC(kernel="linear", C=1, probability=True),
        SVC(gamma=2, C=1, probability=True),
        SVC(kernel="sigmoid", C=1, probability=True)]

    for name, clf in zip(names, classifiers):
        # Tworze dane do nauki i testowania w proporcji
        x_train, y_train, x_test, y_test, unlabel, label = split(
            dataset, 0.1, 0.222)

        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                             ('clf', clf), ])
        text_clf.fit(x_train, y_train)

        # Oceniam dopasowanie
        predicted = text_clf.predict(x_test)
        acc = metrics.accuracy_score(y_test, predicted)
        print(name)
        print("To jest ocena acc przed: ", acc)
        worksheet.write(row, col, name)
        worksheet.write(row, col + 1, acc)
        # Koniec oceny dopasowania

        n_col = 2
        # Tu dzieje sie magia
        for k in range(7):
            worksheet.write(0, n_col, "Wynik " + str(k + 1))
            y_probab = text_clf.predict_proba(unlabel)

            diferences = []

            for i in range(y_probab.shape[0]):
                max = np.amax(y_probab[i])
                diferencesNumber = np.delete(y_probab[i], [np.argmax(y_probab[i])])
                max2 = np.amax(diferencesNumber)
                diference = max - max2
                diferences.append(diference)

            test = []
            diferencesNumber = 0

            for key, value in zip(unlabel, unlabel.index):
                test.append([value, key, diferences[diferencesNumber], diferencesNumber])
                diferencesNumber = diferencesNumber + 1
            test2 = sorted(test, key=lambda x: x[2])

            slaboDopasowane = []
            for i in range(10):
                slaboDopasowane.append(test2[i][0])

            x_train = x_train.append(unlabel[slaboDopasowane])
            y_train = y_train.append(label[slaboDopasowane])

            text_clf.fit(x_train, y_train)

            # Kasowanie najslabiej dopasowanych z label i unlabel
            for i in range(len(slaboDopasowane)):
                unlabel = unlabel.drop(slaboDopasowane[i])
                label = label.drop(slaboDopasowane[i])

            # Oceniam dopasowanie
            predicted = text_clf.predict(x_test)
            acc2 = metrics.accuracy_score(y_test, predicted)
            print("To jest ocena acc po", k, ": ", acc2, "Unlabels: ", len(unlabel), "Train size: ", len(x_train),
                  "Test size: ", len(x_test))
            worksheet.write(row, n_col, acc2)

            # Tu waliduje
            predictedValidation = text_clf.predict(x_validation)
            acc3 = metrics.accuracy_score(y_validation, predictedValidation)
            print("Wynik walidacji: ", acc3, " Wielkosc probki: ", len(x_validation))
            worksheet.write(row + len(names) + 3, n_col, acc3)
            n_col += 1

            # print(k, acc2, len(unlabel), len(x_train), acc3)
        print("\n")
        row += 1
results.close()

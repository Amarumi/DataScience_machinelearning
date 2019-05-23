# -*- coding: utf-8 -*-
# Link do danych: https://www.kaggle.com/iabhishekofficial/mobile-price-classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

pd.set_option("display.max_columns", 10)

dataset=pd.read_csv('train.csv')

X=dataset.drop('price_range',axis=1)
y=dataset['price_range']

print(X.describe())

train_X, val_X, train_y, val_y  = train_test_split(X, y, random_state=101)

model = KNeighborsClassifier(n_neighbors=20)
model.fit(train_X, train_y)

preds_val = model.predict(val_X)
print(classification_report(val_y, preds_val))

matrix = confusion_matrix(val_y, preds_val)
print(matrix)

values = val_y.as_matrix()

# 1. Napiszcie proszę sami kod obliczającą precision, recall oraz f1-score, classification_report().


print(preds_val)
print(values)

def perf_measure(y_actual, y_predict):
    TP0 = 0
    FP0 = 0
    TN0 = 0
    FN0 = 0
    TP1 = 0
    FP1 = 0
    TN1 = 0
    FN1 = 0
    TP2 = 0
    FP2 = 0
    TN2 = 0
    FN2 = 0
    TP3 = 0
    FP3 = 0
    TN3 = 0
    FN3 = 0

    for i in range(len(y_actual)):
        if y_actual[i]==y_predict[i]==0:
            TP0 += 1
        if y_actual[i]!=0 and y_predict[i]==0:
            FP0 += 1
        if y_actual[i]!=y_predict[i] and y_predict[i]==0:
            TN0 += 1
        if y_actual[i]==0 and y_predict[i]!=y_actual[i]:
            FN0 += 1
        if y_actual[i]==y_predict[i]==1:
            TP1 += 1
        if y_actual[i]!=1 and y_predict[i]==1:
            FP1 += 1
        if y_actual[i]!=y_predict[i] and y_predict[i]==1:
            TN1 += 1
        if y_actual[i]==1 and y_predict[i]!=y_actual[i]:
            FN1 += 1
        if y_actual[i]==y_predict[i]==2:
            TP2 += 2
        if y_actual[i]!=2 and y_predict[i]==2:
            FP2 += 2
        if y_actual[i]!=y_predict[i] and y_predict[i]==2:
            TN2 += 2
        if y_actual[i]==2 and y_predict[i]!=y_actual[i]:
            FN2 += 2
        if y_actual[i]==y_predict[i]==3:
            TP3 += 3
        if y_actual[i]!=3 and y_predict[i]==3:
            FP3 += 3
        if y_actual[i]!=y_predict[i] and y_predict[i]==2:
            TN3 += 3
        if y_actual[i]==3 and y_predict[i]!=y_actual[i]:
            FN3 += 3
    return 'prec.0 is',{TP0 / (TP0 ++ FP0)}, 'prec.1 is', {TP1 / (TP1 ++ FP1)}, 'prec.2 is', {TP2 / (TP2 ++ FP2)}, 'prec.3 is', {TP3 / (TP3 ++ FP3)},'rec.0 is', {TP0 / (TP0 ++ FN0)}, 'rec.1 is', {TP1 / (TP1 ++ FN1)}, 'rec.2 is', {TP2 / (TP2 ++ FN2)}, 'rec.3 is', {TP3 / (TP3 ++ FN3)},'f1.0 is',{2*(((TP0 / (TP0 ++ FP0))*(TP0 / (TP0 ++ FN0)))/((TP0 / (TP0 ++ FP0))++(TP0 / (TP0 ++ FN0))))},'f1.1 is',{2*(((TP1 / (TP1 ++ FP1))*(TP1 / (TP1 ++ FN1)))/((TP1 / (TP1 ++ FP1))++(TP1 / (TP1 ++ FN1))))},'f1.2 is',{2*(((TP2 / (TP2 ++ FP2))*(TP2 / (TP2 ++ FN2)))/((TP2 / (TP2 ++ FP2))++(TP2 / (TP2 ++ FN2))))},'f1.3 is',{2*(((TP3 / (TP3 ++ FP3))*(TP3 / (TP3 ++ FN3)))/((TP3 / (TP3 ++ FP3))++(TP3 / (TP3 ++ FN3))))}

print(perf_measure(values, preds_val))

# 2. Napiszcie proszę sami kod obliczającą confusion matrix wartości powinny być takie same jak te które uzyskacie z funkcji confusion_matrix().

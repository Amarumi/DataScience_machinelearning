# coding: utf-8
# https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
import numpy as np
import pandas as pd


df = pd.read_csv('weatherAUS.csv')
print('File shapes:',df.shape)

# Wypisujemy pierwsze 5 wierszy DataFrame.
df[0:5]

# Sprawdźmy które kolumny mają dużo "missing values".
# Żeby dobrze zrozumieć co się dzieje w tej linijce warto wypisać sobie poszczególne
# wartości oddzielnie, czyli:
# df.isnull()
# df.isnull().sum()
df.isnull().sum().sort_values()


# Usuwamy 4 kolumny które mają mniej niż 60 % danych
# Usuwamy też, "location", bo chcemy aby nasz model przewidywał
# dane nizależnie dla całej Australii
# Usuwamy RISK_MM poniważ jest to ściśle powiązane z 'RainTomorrow'
# Na stronie https://www.kaggle.com/jsphyg/weather-dataset-rattle-package możemy przeczytać definicję RISK_MM:
# RISK_MM - The amount of rain. A kind of measure of the "risk".
df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)

# Wypisujemy wymiary "df" po usunięciu 7 kolumn, z 24 zostało ich 17
df.shape


# Usuwamy wszystkie NaN (not a number)
df = df.dropna(how='any')
# Wypisujemy wymiary "df" po usunięciu NaN z 142193 wierszy zostało 112925
df.shape


# Zamieniamy "Yes" i "no" na wartości liczbowe
df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

# Zamiana kategorycznych wartości przy użyciu pd.getDummies()
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
for col in categorical_columns:
    print(np.unique(df[col]))
df = pd.get_dummies(df, columns=categorical_columns)

# Wypisujemy pierwsze 5 wierszy DataFrame. Zwróćmy uwagę na kolumny:
# "WindDir9am_NNW", "WindDir9am_NW", "WindDir9am_S" etc...
# jest to efekt działania funkcji get_dummies(), zamienia ona "stringi" czyli napisy na
# wartości liczbowe. Więcej na:
# https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f
df.iloc[0:5]


# Normalizujemy dane przy użyciu MinMaxScaler()

from sklearn import preprocessing

# W tej linijce tylko tworzymy MinMaxScaler()
scaler = preprocessing.MinMaxScaler()

# Tutaj przekazujemy do Scalera nasze dane i scaler analizuje wszystkie wartości
scaler.fit(df)

# W tej lini następuje faktyczne przekształcenie danych.
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
# Wypisujemy pierwsze 5 wierszy DataFrame. Zwróćmy uwagę na to jak zmieniły się wartości, na przykład "MinTemp".
df.iloc[0:5]


# Tworzymy nasz "input" i "target"
X = df.loc[:,df.columns!='RainTomorrow']
y = df[['RainTomorrow']]



# ------------------- Finding the best model -------------------

from sklearn.model_selection import train_test_split
# Dzielimy dane na train i test set (75% to train set, zaś 25% to test set).
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
print(X_train.shape)
print(X_test.shape)


# ------------------- Logistic Regression -------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Rozpoczynamy pomiar czasu
t0=time.time()

# Tworzymy model LogisticRegression
clf_logreg = LogisticRegression(random_state=0)

# Trenujemy model na danych treningowych
clf_logreg.fit(X_train,y_train)

# Dokonujemy predykcji dla danych testowych.
y_pred = clf_logreg.predict(X_test)

# Obliczamy dokładność dla danych testowych
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)

# Wypisujemy czas który minął
print('Time taken :' , time.time()-t0)



# ------------------- Random Forest Classifier -------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Rozpoczynamy pomiar czasu
t0=time.time()

# Tworzymy model RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=20, max_depth=None, random_state=0, min_samples_leaf=2, criterion="gini")

# Trenujemy model na danych treningowych
clf_rf.fit(X_train,y_train)

# Dokonujemy predykcji dla danych testowych.
y_pred = clf_rf.predict(X_test)

# Obliczamy dokładność dla danych testowych
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)

# Wypisujemy czas który minął
print('Time taken :' , time.time()-t0)



# ------------------- Decision Tree Classifier -------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Rozpoczynamy pomiar czasu
t0=time.time()

# Tworzymy model DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state=0)

# Trenujemy model na danych treningowych
clf_dt.fit(X_train,y_train)

# Dokonujemy predykcji dla danych testowych.
y_pred = clf_dt.predict(X_test)

# Obliczamy dokładność dla danych testowych
score = accuracy_score(y_test,y_pred)
print('Accuracy :',score)

# Wypisujemy czas który minął
print('Time taken :' , time.time()-t0)
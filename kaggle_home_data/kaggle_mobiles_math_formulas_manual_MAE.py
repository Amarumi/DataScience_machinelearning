# -*- coding: utf-8 -*-
# Link do danych: https://www.kaggle.com/c/home-data-for-ml-course
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np

pd.set_option("display.max_columns", 10)
melbourne_data = pd.read_csv("melb_data.csv")
data = melbourne_data.dropna(axis=0)

y = data.Price

feature_list = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = data[feature_list]

print(X.describe())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=101)

model = DecisionTreeRegressor(random_state=101)
model.fit(train_X, train_y)
preds_val = model.predict(val_X)
MAE = mean_absolute_error(val_y, preds_val)
print("(MAE) Mean Absolute Error:  {}".format(MAE))

#################################################################
print()
print('------- ZADANIE 1 - policz MAE ze wzoru --------')

mine_mae = np.mean(abs(val_y - preds_val))
# lub mine_mae = (sum(abs(val_y - preds_val))/val_y.count())
print("(MAE) Manual Mean Absolute Error:  {}".format(mine_mae))

if mine_mae == MAE:
    print('@ @ @ Both manual and function MAE are the same @ @ @')
else:
    print('@ @ @ There is a difference of', mine_mae-mae, '@ @ @')

#################################################################
print()
print('------ ZADANIE 2 - oblicz RMSE z funkcji -------')

#print(val_y[:10])
#print(preds_val[:10])

RMSE = np.sqrt(mean_squared_error(val_y, preds_val))
print("(RMSE) Root Mean Square Error:  {}".format(RMSE))

#################################################################
print()
print('------ ZADANIE 3 - policz RMSE ze wzoru -------')

mine_rmse = np.sqrt(np.mean((abs(val_y - preds_val))**2))
print("(RMSE) Manual Root Mean Square Error:  {}".format(mine_rmse))

if mine_rmse == RMSE:
    print('@ @ @ Both manual and function RMSE are the same @ @ @')
else:
    print('@ @ @ There is a difference of', mine_rmse-RMSE, '@ @ @')

#################################################################
print()
print('-- ZADANIE 4 - policz sr. cene nieruchomosci --')
# 4. Napiszcie prosty "model" zawsze zwracający średnią cenę nieruchomości
# Sprawdzcie jakie są wartości MAE i RMSE dla takiego modelu. Czy jest on lepszy niż DecisionTreeRegressor()?

"""y = data.Price

features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=101)

for my_features in [1,2,3,4,5,6,7]:
    model_avg = DecisionTreeRegressor(max_features=my_features, random_state=101)
    model_avg.fit(train_X, train_y)
    preds_val = model_avg.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)
    rmse = np.sqrt(mean_squared_error(val_y, preds_val))
    print("Max features: {}    MSE:  {}    RMSE:  {}".format(my_features, mae, rmse))
"""

#################################################################
print()
print('-- ZADANIE 4 - policz medaine cene nieruchomosci --')
# 5. Napiszcie prosty "model" zawsze zwracający medianę cen nieruchomości.
# Sprawdzcie jakie są wartości MAE i RMSE dla takiego modelu. Czy jest on lepszy niż DecisionTreeRegressor()?

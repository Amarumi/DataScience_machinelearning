# -*- coding: utf-8 -*-
"""
Created on 2nd Mar'19
@ author: Justyna
"""

from sklearn.datasets import load_diabetes
import pickle
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
from sklearn.model_selection import cross_val_score, KFold

## ----------------- create dataset -----------------
digits = load_diabetes()
features = digits.data
target = digits.target


## ---------------- dataset training ----------------
print('______________________________________________________________________')
print("Dataset before training split:{}".format(digits.data.shape))

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25)
print('______________________________________________________________________')
print("Dataset after training split | Train:{} Test:{}". format(X_train.data.shape, X_test.data.shape))


## ----------- sklearn.impute.SimpleImputer ---------
my_imputer = Imputer()
train_X = my_imputer.fit_transform(X_train)
test_X = my_imputer.transform(X_test)


## ----------------- XGBRegressor -----------------
df_cv = pd.DataFrame(columns=['max_depth','learning_rate','mean_squared_error'])

for d in [3,4,5,6,7,8]:
    for r in np.arange(0.05, 0.35, 0.05):
        clf_xgbr = XGBRegressor(max_depth=d, learning_rate=r, n_estimators=10, silent=True, objective='reg:linear', booster='gbtree',
                            n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                            colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0,
                            seed=None, missing=None, importance_type='gain')

        # ----------------------- cross validation -----------------------
        scorer = make_scorer(mean_squared_error)
        kfold = KFold(n_splits=20, random_state=11)
        results = cross_val_score(clf_xgbr, train_X, y_train, cv=kfold, scoring=scorer)

        # ----------------------- results dataframe ----------------------
        for ind in range(len(results)):
            rows = [d, r, results[ind]]
            df_tmp = pd.DataFrame(data=[rows], columns=list(df_cv.columns.values))
            df_cv = df_cv.append(df_tmp, ignore_index=True)


## ------------- find min/max error --------------
min_mse = df_cv['mean_squared_error'].min()
max_mse = df_cv['mean_squared_error'].max()
print('Minimum_MSE:{}, Maximum_MSE:{}'.format(min_mse, max_mse))


## ---------------- store results ----------------
all_in_one = {'XGBRegressor_Results:': df_cv, 'Minimum_MSE:': min_mse, 'Maximum_MSE:': max_mse}

with open('/home/justyna/Repo/jdsz2-homeworks/python8/justyna_krygier/results.pickle', 'wb') as f:
    pickle.dump(all_in_one, f)
#
# sys.stdout = open('results', 'a')
# print(all_in_one)
# sys.stdout.close()

#print(df_cv['mean_squared_error'].isna().any())
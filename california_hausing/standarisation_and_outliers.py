from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import BallTree
import pickle


## --------- create dataframe, add .target ----------

dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data ,columns = dataset.feature_names)
df['target'] = pd.Series(dataset.target)

## --------- 2 features chosen for training ---------

X = df[['Population','HouseAge']]
print('_____________________________________')
print('Dataset before train split')
print(X.describe())
y = df.target

## ------------- split for train & test -------------

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)
df2 = pd.DataFrame(data=X_train, columns=['Population','HouseAge'])

## ---------------- feature distance ----------------

plt.title('California Housing')
plt.scatter(df2['Population'],df2['HouseAge'], color = 'darkcyan', marker = '.')
plt.xlabel('Population')
plt.ylabel('House Age')
plt.show()

## ---------- standarisation with outliers ----------

print('_____________________________________')
print('Dataset after train split')
print(df2.describe())
norm_scale = preprocessing.StandardScaler().fit(df2[['Population', 'HouseAge']])
df2_norm = norm_scale.transform(df2[['Population', 'HouseAge']])

plt.figure(figsize=(10,10))
plt.title('California Housing')
plt.scatter(df2_norm[:,0], df2_norm[:,1], color='darkorchid', alpha=0.5)
plt.xlabel('Population')
plt.ylabel('House Age')
plt.show()

## --------------- removing outliers ----------------

Q1 = df2['Population'].quantile(0.25)
Q3 = df2['Population'].quantile(0.75)
IQR = Q3 - Q1

down = float(Q1 - 1.5 * IQR)
up = float(Q3  + 1.5 * IQR)

df2_ro = df2.loc[(df2['Population'] > down) & (df2['Population'] < up)]

## ------- transformation - removed outliers --------

norm_scale = preprocessing.StandardScaler().fit(df2_ro[['Population', 'HouseAge']])
df2_norm_ro = norm_scale.transform(df2_ro[['Population', 'HouseAge']])

plt.figure(figsize=(10,10))
plt.title('California Housing - outliers removed')
plt.scatter(df2_norm_ro[:,0], df2_norm_ro[:,1], color='darkorchid', alpha=0.3)
plt.xlabel('Population')
plt.ylabel('House Age')
plt.show()

## ------------------- statistics ------------------

population_sd = np.std(df['Population'])
population_avg = np.mean(df['Population'])
hage_sd = np.std(df['HouseAge'])
hage_avg = np.mean(df['HouseAge'])

print('_____________________________________')
print('Parameters used for standarisation')
statistics = {'Population': {'Average': population_avg,
                            'Standard deviation': population_sd},
              'HouseAge': {'Average':hage_avg,
                           'Standard deviation': hage_sd}}
print(statistics)

## ------------------- ball tree ------------------

X = df2_norm_ro
tree = BallTree(X[1:], leaf_size=40,metric ='euclidean' )

## --------------------- pickle -------------------

model_obj = {'model':tree,'statistics':statistics }
print(model_obj)

with open('model.pickle', 'wb') as f:
    pickle.dump(model_obj, f)

with open('model.pickle', 'rb') as f:
    loaded_model = pickle.load(f)

loaded_tree = loaded_model['model']
dist, ind = loaded_tree.query(X[0].reshape(1,2), k=5)

## ------------- distance K neighbors --------------

print('distance k ngh \n',dist,'\n')
print('index k ngh  \n',ind,'\n')
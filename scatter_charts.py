from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from statsmodels.formula.api import ols
import numpy as np

#diabets = datasets.load_diabets()
#boston = datasets.load_boston()

def makedf(dataset):
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = pd.Series(dataset.target)
    return df

df = makedf(datasets.load_boston())
ds_name = 'boston'
print(df.info())
print(df.describe())

#df = makedf(datasets.load_boston())
#ds_name = 'boston'
#print(df.info())
#print(df.describe())

for row, col in enumerate(df.columns[0:-1]):

    x = df[col]
    y = df[df.columns[-1]]

    LR = ols("x ~ y", df).fit()
    print("\n[!] ", col)
    print(LR.summary().tables[1])
    print()

    LR = linear_model.LinearRegression()
    X = x[:, np.newaxis]
    Y = y[:, np.newaxis]
    LR.fit(X, Y)

    plt.scatter(x, y, color = 'DarkGreen', marker = '.')
    plt.plot(X, LR.coef_ * X + LR.intercept_, linewidth=1)
    plt.title(ds_name)
    plt.ylabel(df.columns[-1])
    plt.xlabel(col)
    plt.show()



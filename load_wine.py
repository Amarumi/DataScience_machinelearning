from sklearn.datasets import load_wine
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from statsmodels.formula.api import ols
import numpy as np

data = datasets.load_wine()
data_name = 'wine'
print(data)

x =
y =

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=101)
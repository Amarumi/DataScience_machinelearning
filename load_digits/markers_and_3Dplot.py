# -*- coding: utf-8 -*-
"""
Created on 18th Feb'19
@ author: Justyna
"""

from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import manifold
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans



## --------- create dataset, add .target -----------
digits = load_digits()

#plt.gray()
#plt.matshow(digits.images[0])
#plt.show()

features = digits.data
target = digits.target


## ----------------- create training ----------------
print('______________________________________________________________________')
print("Dataset before training split :", digits.data.shape)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)
print('______________________________________________________________________')
print("Dataset after training split", "| Train:", X_train.data.shape, "| Test: ",X_test.data.shape)


## ------------------ create model ------------------
clf = OneVsRestClassifier(svm.SVC(kernel='linear'))
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("accuracy: ", acc)

cm = confusion_matrix(y_test, y_pred)
print("conf_matrix: ")
print(cm)

## ----------------- visualisation 2D ------------------
tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
X = tsne.fit_transform(X_test)

color = ["#"+''.join([random.choice('0123456789ABCDEF')
         for j in range(6)])
         for i in range(len(digits.target_names))]

plt.figure(figsize=(10,10))

for i in range(len(digits.target_names)):
    plt.scatter(x= X[y_test == i, 0], y= X[y_test == i, 1], c=color[i], s = 40)
    plt.legend(set(list(target)))

# plot missed predictions
plt.scatter(x= X[y_test!=y_pred, 0], y= X[y_test!=y_pred, 1], c='red', s=80, marker='x')
plt.title('Load_digits with SVM OneVsRest model, t-SNE: 2d')
plt.show()

## ----------------- visualisation 3D ------------------
yext = np.expand_dims(y_test, 1)
Xext = np.concatenate((X, yext), 1)

fig = plt.figure()
threeDFig = fig.add_subplot(111, projection='3d')

# HOW TO ASSIGN SAME COLOR PALETTE ????
threeDFig.scatter3D(Xext[:, 0], Xext[:, 1], y_test, c=y_test, cmap='autumn')

threeDFig.set_xlabel('x')
threeDFig.set_ylabel('y')
threeDFig.set_zlabel('r')
threeDFig.set_title('Load_digits with SVM OneVsRest model, t-SNE: 3d')

#print("clf.coef_: ", clf.coef_)
#print("clf.intercept_: ", clf.intercept_)

tmp = np.linspace(-1, 1, 21)
xSpace, ySpace = np.meshgrid(tmp, tmp)
zSpace = (-clf.intercept_[0]-clf.coef_[0][0]*xSpace-clf.coef_[0][1]*ySpace)

threeDFig.plot_surface(xSpace, ySpace, zSpace)
plt.show()

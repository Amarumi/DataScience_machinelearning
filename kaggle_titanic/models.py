import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors, datasets
from sklearn.ensemble import RandomForestClassifier
import csv
import sys

pd.set_option('display.max_columns', 50)

# ---------- Create train and test samples ----------
train_df = pd.read_csv('train.csv')
train_df['Sex'].replace({'female': 1, 'male': 0},inplace = True)
train_df['Age'].fillna(value = train_df['Age'].mean(), inplace = True)

X_train = train_df[['Sex','Pclass','Age']]
y_train = train_df['Survived']
#print(train_df.head())
#print(train_df.isnull().any())

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


test_df = pd.read_csv('test.csv')
test_df['Sex'].replace({'female': 1, 'male': 0},inplace = True)

columns_to_delete = ['PassengerId','Survived','Name','Ticket']
categorical_columns = ['Sex','Fare','Cabin','Embarked']
target = train_df['Survived']
train_df = train_df.drop(columns=columns_to_delete, axis=1)

# create features for unique records, ex. Sex - split for male and female
train_df = pd.get_dummies(train_df, columns=categorical_columns)

test_df['Age'].fillna(value = test_df['Age'].mean(), inplace = True)

X_test = test_df[['Sex','Pclass','Age']]


# --------------- SVM ---------------

clf_SVM = svm.SVC(kernel='rbf', C=1, gamma='scale')
clf_SVM.fit(X_train, y_train)
y_SVM = clf_SVM.predict(X_test)

SVM = pd.DataFrame(test_df['PassengerId'], columns=['PassengerId'])
SVM.loc[:,'Survived'] = y_SVM
SVM.to_csv('svm_submission', index=False)

print('SVM public score: 0.69856')

# acc_SVM = accuracy_score(y_test, y_SVM)
# print("acc SVM: ", acc_SVM)
# cm_SVM = confusion_matrix(y_test, y_SVM)
# print(cm_SVM)
# print('-----------------------------------------------------')


# ------------- LOG REG -------------

clf_LR = LogisticRegression(random_state=0, solver='lbfgs', max_iter=400)
clf_LR.fit(X_train, y_train)
y_LR = clf_LR.predict(X_test)

LR = pd.DataFrame(test_df['PassengerId'], columns=['PassengerId'])
LR.loc[:,'Survived'] = y_LR
LR.to_csv('lr_submission', index=False)

print('Logistic Regression public score: 0.75598')

# acc_LR = accuracy_score(y_test, y_LR)
# print("acc Logistic Regression: ", acc_LR)
# cm_LR = confusion_matrix(y_test, y_LR)
# print(cm_LR)
# print('-----------------------------------------------------')


# --------------- BAYES ---------------

clf_GNB = GaussianNB()
clf_GNB.fit(X_train, y_train)
y_GNB = clf_GNB.predict(X_test)

GNB = pd.DataFrame(test_df['PassengerId'], columns=['PassengerId'])
GNB.loc[:,'Survived'] = y_GNB
GNB.to_csv('gnb_submission', index=False)

print('Gaussian public score: 0.77033')

# acc_GNB = accuracy_score(y_test, y_GNB)
# print("acc Bayes: ", acc_GNB)
# cm_GNB = confusion_matrix(y_test, y_GNB)
# print(cm_GNB)
# print('-----------------------------------------------------')


# --------------- Random Forest ---------------

clf_RF = RandomForestClassifier(n_estimators=10, max_depth=10,random_state=0, min_samples_leaf=2, criterion="gini")
clf_RF.fit(X_train, y_train)
y_RF = clf_RF.predict(X_test)

RF = pd.DataFrame(test_df['PassengerId'], columns=['PassengerId'])
RF.loc[:,'Survived'] = y_RF
RF.to_csv('ranfor_submission', index=False)

print('Random Forest public score: 0.76076')

# acc_RF = accuracy_score(y_test, y_RF)
# print("acc Random Forest: ", acc_RF)
# cm_RF = confusion_matrix(y_test, y_RF)
# print(cm_RF)
# print('-----------------------------------------------------')


# --------------- KNN ---------------

n_neighbors = 3
clf_KNN = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf_KNN.fit(X_train, y_train)
y_KNN = clf_KNN.predict(X_test)

KNN = pd.DataFrame(test_df['PassengerId'], columns=['PassengerId'])
KNN.loc[:,'Survived'] = y_KNN
KNN.to_csv('knn_submission', index=False)

print('Logistic Regression public score: 0.72248')

# acc_KNN = accuracy_score(y_test, y_pred)
# print("acc Random Forest: ", acc_KNN)
# cm_KNN = confusion_matrix(y_test, y_pred)
# print(cm_KNN)
# print('-----------------------------------------------------')

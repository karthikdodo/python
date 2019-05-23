#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:08:56 2019

@author: karthikchowdary
"""

import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")


train = pd.read_csv('/Users/karthikchowdary/Desktop/KarthiK/graduate/spring-19/python/lab-2/train.csv')



train.isna().head()




train.fillna(train.mean(), inplace=True)



train.info(),
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(train['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])


train.info()


#caluculating the svm
feature_cols = ['PassengerId' ,'Pclass', 'Age' ,'SibSp' ,'Parch' ,'Fare' , 'Sex']
from sklearn.svm import SVC

X = train[feature_cols]
y = train.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

svm = SVC()                        
expected = train.Survived

svm.fit(X_train, y_train)


predicted_label = svm.predict(train[feature_cols])          



print(metrics.confusion_matrix(expected, predicted_label))

# Cross Validation compare the predicted and expected values
print(metrics.classification_report(expected, predicted_label))


print('Accuracy of SVM classifier : {:.2f}'
     .format(svm.score(X_test, y_test)))


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()      
model.fit(train[feature_cols], train.Survived)

expected = train.Survived            
predicted = model.predict(train[feature_cols])


print(metrics.classification_report(expected, predicted))    

print(metrics.confusion_matrix(expected, predicted))         
X_train, X_test, Y_train, Y_test = train_test_split(train[feature_cols], train.Survived, test_size=0.2, random_state=0)

model.fit(X_train, Y_train)                                 

Y_predicted = model.predict(X_test)                           

print("accuracy using Gaussian navie bayes Model is ", metrics.accuracy_score(Y_test, Y_predicted) * 100)



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()


logreg.fit(X, y)

logreg.predict(X)
y_pred = logreg.predict(X)
len(y_pred)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print("accuracy using knn Model is ",metrics.accuracy_score(y, y_pred))

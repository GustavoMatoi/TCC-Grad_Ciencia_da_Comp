import os
import numpy as np
import cv2
import matplotlib as plt 
import csv
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import xgboost
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
import csv
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

dir = 'C://Users//gutei//tcc'

categories = ['to\\artigo', 'saudavel\\artigo']

labels = [1,0]
data = []

for category in categories:
    path = os.path.join(dir, category)
    label = labels[categories.index(category)]  # Index is used to find the corresponding label
    for image in os.listdir(path):
        imgPath = os.path.join(path, image)
        img = cv2.imread(imgPath, 0)
        try:
            img = cv2.resize(img, (50,50))
            img = np.array(img).flatten()
            data.append([img, label])
        except Exception as e:
            pass

pick_in = open('data1.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

features = []
labels = []

for feature, label in data: 
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.30)

clf = svm.SVC(kernel="linear", probability=True)
#scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
scores = cross_validate(clf, xtest, ytest, cv = 10, scoring=['accuracy','f1','precision','recall','roc_auc'])

print('Linear SVM ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))
y_pred = cross_val_predict(clf, xtest, ytest, cv=10, n_jobs=-1)

clf = svm.SVC(kernel="rbf", probability=True)
scores = cross_validate(clf,xtest, ytest, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Radial SVM ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))


clf = svm.SVC(kernel="poly", probability=True)
scores = cross_validate(clf, xtest, ytest, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Poly SVM ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))


clf = KNeighborsClassifier(60)
scores = cross_validate(clf, xtest, ytest, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('KNN ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))


clf = DecisionTreeClassifier()
scores = cross_validate(clf, xtest, ytest, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('DT ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))


clf = RandomForestClassifier(n_estimators=100)
scores = cross_validate(clf, xtest, ytest, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('RF ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))


clf = MLPClassifier(max_iter=5000)
scores = cross_validate(clf, xtest, ytest, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('MLP ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))


clf = AdaBoostClassifier()
scores = cross_validate(clf,xtest, ytest, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Adaboost ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))


clf = QuadraticDiscriminantAnalysis()
scores = cross_validate(clf, xtest, ytest, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('QDA ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))


clf = GaussianNB()
scores = cross_validate(clf,xtest, ytest, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('NB ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))


clf = SGDClassifier()
scores = cross_validate(clf, xtest, ytest, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('SGD ', 
      format(np.mean(scores['test_accuracy']), ".2f"),
      format(np.mean(scores['test_f1']), ".2f"),
      format(np.mean(scores['test_precision']), ".2f"),
      format(np.mean(scores['test_recall']), ".2f"),
      format(np.mean(scores['test_roc_auc']), ".2f"))

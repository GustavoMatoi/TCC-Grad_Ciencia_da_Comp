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
from sklearn.model_selection import cross_validate, cross_val_predict
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
import time

# Carrega os dados
dataset = pd.read_csv("refatoracaoV3.csv")
X = dataset.iloc[:, 0:219].values
y = dataset.iloc[:, 219].values
y = y.astype('float32')

minimax = MinMaxScaler()
X = minimax.fit_transform(X)

KBest = SelectKBest(chi2, k=100).fit(X, y)
f = KBest.get_support(1)
# Normaliza a base de dados
minimax = MinMaxScaler()
X = minimax.fit_transform(X)
X = SelectKBest(chi2, k=100).fit_transform(X, y)
print(y[0])


print('Resultado do experimento')
print('Classifier, Accuracy, F1, Precision, Recall, AUC')
inicio = time.time()

clf = xgboost.XGBClassifier()
scores = cross_validate(clf, X, y, cv = 10, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")
print('XGBoost ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))


clf = Perceptron(tol=1e-3, random_state=0)
scores = cross_validate(clf, X, y, cv = 10, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")
print('Perceptron ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))


inicio = time.time()

clf = svm.SVC(kernel="linear", probability=True)
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Linear SVM ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")

inicio = time.time()

clf = svm.SVC(kernel="sigmoid", probability=True)
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Sigmoid SVM ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))

fim = time.time()

print(f"Tempo de execução: {fim - inicio} segundos")

inicio = time.time()

clf = svm.SVC(kernel="rbf", probability=True)
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Radial SVM ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")

inicio = time.time()

clf = svm.SVC(kernel="poly", probability=True)
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Polynomial SVM ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))




fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")
y_pred = cross_val_predict(clf, X, y, cv=10, n_jobs=-1)

# Calcular a matriz de confusão
matrizConfusao = confusion_matrix(y, y_pred)

# Imprimir a matriz de confusão
print('Matriz de Confusão:\n', matrizConfusao)
inicio = time.time()

clf = KNeighborsClassifier(60)
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('KNN ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")

inicio = time.time()

clf = DecisionTreeClassifier()
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Decision Tree ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")


inicio = time.time()

clf = RandomForestClassifier()
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Random Forest ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")

inicio = time.time()

clf = MLPClassifier(max_iter=5000)
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('MLP ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")
y_pred = cross_val_predict(clf, X, y, cv=10, n_jobs=-1)

# Calcular a matriz de confusão
matrizConfusao = confusion_matrix(y, y_pred)
print(matrizConfusao)



inicio = time.time()

clf = AdaBoostClassifier()
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Adaboost Classifier ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")

inicio = time.time()

clf = QuadraticDiscriminantAnalysis()
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('QDA ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")

inicio = time.time()

clf = GaussianNB()
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('Naive Bayes ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")

inicio = time.time()

clf = SGDClassifier()
scores = cross_validate(clf, X, y, cv = 10, n_jobs=-1, scoring=['accuracy','f1','precision','recall','roc_auc'])
#print('Resultado XGBoost por fold')
#print(scores)
print('SGD ',np.mean(scores['test_accuracy']),np.mean(scores['test_f1']),np.mean(scores['test_precision']),np.mean(scores['test_recall']),np.mean(scores['test_roc_auc']))
fim = time.time()
print(f"Tempo de execução: {fim - inicio} segundos")
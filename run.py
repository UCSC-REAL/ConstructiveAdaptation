import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
import argparse
EPS = np.finfo(float).eps

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult', help='Dataset to run.')
args = parser.parse_args()

# load dataset
if args.dataset.lower() == 'credit':
    exec(open("credit.py").read())
elif args.dataset.lower() == 'german':
    exec(open("german.py").read())
elif args.dataset.lower() == 'spam':
    exec(open("spam.py").read())
else:
    exec(open("adult.py").read())

# cost matrix
s_I, s_M = 1, 5
S = np.identity(N_I + N_M)
for i in range(N_I):
    S[i][i] = s_I
for i in range(N_M):
    S[i+N_I][i+N_I] = s_M

# load the baseline methods
exec(open("LightTouch.py").read())

clf = LogisticRegression(fit_intercept=True).fit(x, y)
print((clf.predict(x)==y).mean())

skf = StratifiedKFold(n_splits=5)
print("-------------------Static---------------------")
accuracy = []
fraction = []
test_acc = []
deterioration = []
for train, test in skf.split(x, y):
    X_train, y_train, X_test, y_test = x[train], y[train], x[test], y[test]
    acc0, acc, frac, dete = run_svm(X_train, y_train, X_test, y_test)
    accuracy.append(acc)
    fraction.append(frac)
    deterioration.append(dete)
    test_acc.append(acc0)
test_acc = np.array(test_acc)
accuracy = np.array(accuracy)
fraction = np.array(fraction)
deterioration = np.array(deterioration)
print(f"${test_acc.mean()*100:2.2f}$ \\\\ ${accuracy.mean()*100:2.2f}$ \\\\ ${fraction.mean()*100:2.2f}$ \\\\ ${deterioration.mean()*100:2.2f}$")
print(f"${test_acc.mean()*100:2.2f}\pm{test_acc.std()*100:.2f}$ \\\\ ${accuracy.mean()*100:2.2f}\pm{accuracy.std()*100:.2f}$ \\\\ ${fraction.mean()*100:2.2f}\pm{fraction.std()*100:.2f}$ \\\\ ${deterioration.mean()*100:2.2f}\pm{deterioration.std()*100:.2f}$")

print("-------------------DropFeatures---------------------")
test_acc = []
accuracy = []
fraction = []
deterioration = []
for train, test in skf.split(x, y):
    X_train, y_train, X_test, y_test = x_I[train], y[train], x_I[test], y[test]
    acc0, acc, frac, dete = run_improvable_svm(X_train, y_train, X_test, y_test)
    test_acc.append(acc0)
    accuracy.append(acc)
    fraction.append(frac)
    deterioration.append(dete)
test_acc = np.array(test_acc)
accuracy = np.array(accuracy)
fraction = np.array(fraction)
deterioration = np.array(deterioration)
print(f"${test_acc.mean()*100:2.2f}$ \\\\ ${accuracy.mean()*100:2.2f}$ \\\\ ${fraction.mean()*100:2.2f}$ \\\\ ${deterioration.mean()*100:2.2f}$")
print(f"${test_acc.mean()*100:2.2f}\pm{test_acc.std()*100:.2f}$ \\\\ ${accuracy.mean()*100:2.2f}\pm{accuracy.std()*100:.2f}$ \\\\ ${fraction.mean()*100:2.2f}\pm{fraction.std()*100:.2f}$ \\\\ ${deterioration.mean()*100:2.2f}\pm{deterioration.std()*100:.2f}$")

print("-------------------ManipulationProof---------------------")
test_acc = []
accuracy = []
fraction = []
deterioration = []
for train, test in skf.split(x, y):
    X_train, y_train, X_test, y_test = x[train], y[train], x[test], y[test]
    acc0, acc, frac, dete = run_strategic_clf(X_train, y_train, X_test, y_test)
    accuracy.append(acc)
    fraction.append(frac)
    deterioration.append(dete)
    test_acc.append(acc0)
test_acc = np.array(test_acc)
accuracy = np.array(accuracy)
fraction = np.array(fraction)
deterioration = np.array(deterioration)
print(f"${test_acc.mean()*100:2.2f}$ \\\\ ${accuracy.mean()*100:2.2f}$ \\\\ ${fraction.mean()*100:2.2f}$ \\\\ ${deterioration.mean()*100:2.2f}$")
print(f"${test_acc.mean()*100:2.2f}\pm{test_acc.std()*100:.2f}$ \\\\ ${accuracy.mean()*100:2.2f}\pm{accuracy.std()*100:.2f}$ \\\\ ${fraction.mean()*100:2.2f}\pm{fraction.std()*100:.2f}$ \\\\ ${deterioration.mean()*100:2.2f}\pm{deterioration.std()*100:.2f}$")

# print(f"${accuracy.mean()*100:2.2f}\pm{accuracy.std()*100:.2f}$ \\\\ ${fraction.mean()*100:2.2f}\pm{fraction.std()*100:.2f}$ \\\\ ${deterioration.mean()*100:2.2f}\pm{deterioration.std()*100:.2f}$")

print("-------------------LightTouch---------------------")
# lbds = np.arange(0., 1.5, 0.1)
lbds = [1.]
for l in lbds:
    accuracy = []
    test_acc = []
    fraction = []
    deterioration = []
    for train, test in skf.split(x, y):
        X_train, y_train, X_test, y_test = x[train], y[train], x[test], y[test]
        acc0, acc, frac, dete = run_recourse_clf(X_train, y_train, X_test, y_test, lbd=l)
        test_acc.append(acc0)
        accuracy.append(acc)
        fraction.append(frac)
        deterioration.append(dete)
    test_acc = np.array(test_acc)
    accuracy = np.array(accuracy)
    fraction = np.array(fraction)
    deterioration = np.array(deterioration)
    print(f"${test_acc.mean()*100:2.2f}$ \\\\ ${accuracy.mean()*100:2.2f}$ \\\\ ${fraction.mean()*100:2.2f}$ \\\\ ${deterioration.mean()*100:2.2f}$")
    print(f"${test_acc.mean()*100:2.2f}\pm{test_acc.std()*100:.2f}$ \\\\ ${accuracy.mean()*100:2.2f}\pm{accuracy.std()*100:.2f}$ \\\\ ${fraction.mean()*100:2.2f}\pm{fraction.std()*100:.2f}$ \\\\ ${deterioration.mean()*100:2.2f}\pm{deterioration.std()*100:.2f}$")

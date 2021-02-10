import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold

EPS = np.finfo(float).eps

from responsibly.dataset import AdultDataset

def run_svm(X_train, y_train, X_test, y_test):
    svm = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
    y1 = svm.predict(X_test)
    acc0 = (y1 == y_test).mean()
    weight = np.concatenate((svm.coef_.squeeze(), svm.intercept_))
    delta_M = best_response_M(weight, X_test)
    y_M = svm.predict(delta_M)
    acc = (y_test == y_M).mean()
    # print("Accuracy: ", (y.values == y_M).mean())
    delta_I = best_response_I(weight, X_test)
    y_I = svm.predict(delta_I)
    frac = (y_I[y_test <= 0] > 0).mean()
    dete = (y_I[y_test > 0] <= 0).mean()
    # print("recourse fraction: ", (y_I[y.values <= 0] > 0).mean())
    return acc0, acc, frac, dete

def run_improvable_svm(X_train, y_train, X_test, y_test):
    svm = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
    weight = np.concatenate((svm.coef_.squeeze(), svm.intercept_))
    acc = (svm.predict(X_test) == y_test).mean()
    # print("Accuracy: ", (y.values == y_M).mean())
    delta_I = best_response_I(weight, X_test)
    y_I = svm.predict(delta_I)
    frac = (y_I[y_test <= 0] > 0).mean()
    dete = (y_I[y_test > 0] <= 0).mean()
    # print("recourse fraction: ", (y_I[y.values <= 0] > 0).mean())
    return acc, acc, frac, dete

def run_strategic_clf(X_train, y_train, X_test, y_test):
    res = minimize(strategic_objective(X_train, y_train), np.concatenate((clf.coef_.squeeze(), clf.intercept_)), method='L-BFGS-B')
    y1 = np.where(X_test @ res.x[:-1] + res.x[-1] >= 0, 1, -1)
    acc0 = (y_test == y1).mean()    
    delta_M = best_response_M(res.x, X_test)
    y_M = np.where(delta_M @ res.x[:-1] + res.x[-1] > 0, 1, -1)
    acc = (y_test == y_M).mean()
    # print("Accuracy: ", (y.values == y_M).mean())
    delta_I = best_response_I(res.x, X_test)
    y_I = np.where(delta_I @ res.x[:-1] + res.x[-1] > 0, 1, -1)
    frac = (y_I[y_test < 0] >= 0).mean()
    dete = (y_I[y_test >= 0] < 0).mean()
    # print("recourse fraction: ", (y_I[y.values <= 0] > 0).mean())
    return acc0, acc, frac, dete

def run_recourse_clf(X_train, y_train, X_test, y_test, lbd=1.0):
    res = minimize(objective(X_train, y_train, lbd=lbd), np.concatenate((clf.coef_.squeeze(), clf.intercept_)), method='L-BFGS-B')
    y1 = np.where(X_test @ res.x[:-1] + res.x[-1] >= 0, 1, -1)
    acc0 = (y1 == y_test).mean()
    delta_M = best_response_M(res.x, X_test)
    y_M = np.where(delta_M @ res.x[:-1] + res.x[-1] >= 0, 1, -1)
    acc = (y_test == y_M).mean()
    delta_I = best_response_I(res.x, X_test)
    y_I = np.where(delta_I @ res.x[:-1] + res.x[-1] >= 0, 1, -1)
    frac = (y_I[y_test < 0] >= 0).mean()
    dete = (y_I[y_test >=0] < 0).mean()
    return acc0, acc, frac, dete

def objective(x, y, lbd):
    def f(w):
        # output of linear classifier
        out = x @ w[:-1] + w[-1]
        # temporal variables of denominator
        dnm_I = np.sqrt(s_I * np.dot(w[:N_I], w[:N_I])).item()
        dnm_M = np.sqrt(s_M * np.dot(w[N_I:N_I+N_M], w[N_I:N_I + N_M])).item()
        # loss for two terms
        loss_I = (1. / (1. + np.exp(-out - 2 * dnm_I + EPS)))
        loss_M = (1. / (1. + np.exp(-out - 2 * dnm_M + EPS)))
        
        loss = (2 * loss_M - 1) * y + lbd * loss_I - 0.5 * w.transpose() @ w
        return -loss.mean()
    return f

def strategic_objective(x, y):
    def f(w):
        # output of linear classifier
        out = x @ w[:-1] + w[-1]
        # temporal variables of denominator
        dnm_M = np.sqrt(s_I * np.dot(w[:N_I], w[:N_I]) + s_M * np.dot(w[N_I:N_I+N_M], w[N_I:N_I + N_M])).item()
        # loss for two terms
        loss_M = (1. / (1. + np.exp(-out - 2 * dnm_M + EPS)))
        
        loss = (2 * loss_M - 1) * y  - 0.5 * np.dot(w, w)
        return -loss.mean()
    return f

def best_response(coef, arr):
    w = coef[:-1]
    w_f = coef[:N_I+N_M]
    b = coef[-1]
    tmp = np.array((S @ w_f) / np.array(w_f.transpose() @ S @ w_f)).squeeze()
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        t = np.copy(arr[i])
        if np.dot(w, t) + b < 0 and abs(np.dot(w, t) + b) < np.array(np.sqrt(w_f.transpose() @ S @ w_f)).squeeze():
            t[:N_I + N_M] = t[:N_I + N_M] - (np.dot(w, t) + b) * tmp
        out[i] = t
    return out

def best_response_I(coef, arr):
    w = coef[:-1]
    w_f = coef[:N_I]
    b = coef[-1]
    tmp = np.array((S[:N_I, :N_I] @ w_f) / np.array(w_f.transpose() @ S[:N_I, :N_I] @ w_f)).squeeze()
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        t = np.copy(arr[i])
        if np.dot(w, t) + b < 0 and abs(np.dot(w, t) + b) < np.array(np.sqrt(w_f.transpose() @ S[:N_I, :N_I] @ w_f)).squeeze():
            t[:N_I] = t[:N_I] - (np.dot(w, t) + b) * tmp
        out[i] = t
    return out

def best_response_M(coef, arr):
    w = coef[:-1]
    w_f = coef[N_I:N_I+N_M]
    b = coef[-1]
    tmp = np.array((S[N_I:, N_I:] @ w_f) / np.array(w_f.transpose() @ S[N_I:, N_I:] @ w_f)).squeeze()
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        t = np.copy(arr[i])
        if np.dot(w, t) + b < 0 and abs(np.dot(w, t) + b) < np.array(np.sqrt(w_f.transpose() @ S[N_I:, N_I:] @ w_f)).squeeze():
            t[N_I:N_I+N_M] = t[N_I:N_I+N_M] - (np.dot(w, t) + b) * tmp
        out[i] = t
    return out
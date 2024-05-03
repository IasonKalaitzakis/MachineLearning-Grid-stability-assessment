# evaluate RFE for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense

from datetime import datetime

# create dataset
data = pd.read_csv('smart_grid_stability_augmented.csv')
map1 = {'unstable': 0, 'stable': 1}
data['stabf'] = data['stabf'].replace(map1)
data = data.sample(frac=1)

X,z,y=np.hsplit(data, np.array([12,13]))

# create pipeline
rfe = RFECV(estimator=DecisionTreeClassifier())
#rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=4)
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)\n' % (mean(n_scores), std(n_scores)))

rfe.fit(X, y)
for i in range(X.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i+1, rfe.support_[i], rfe.ranking_[i]))
print('\n')


